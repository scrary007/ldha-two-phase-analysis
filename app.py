"""LDHA biphasic NADH time-course analysis: standalone Streamlit dashboard.

Usage: streamlit run app.py
No cloud database, file storage, or Google credentials are required.
"""
from __future__ import annotations

from hashlib import sha1
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analysis import (
    DataError, apparent_postmix_fraction, classify_timecourse, filename_pyruvate, fit_all_models, fit_line,
    fit_substrate_inhibition, read_cary_csv, recommended_windows,
    signal_slope_to_velocity, substrate_inhibition,
)
from report import make_pdf, make_all_traces_pdf, make_acceptance_pdf
from acceptance import (SINGLE, EXPONENTIAL, EARLY, EXCLUDE, METHODS, suggest_method,
                        NOT_REVIEWED, KINETIC, DESCRIPTIVE, EXCLUDED,
                        available_decisions, recommended_decision,
                        analysis_fingerprint, compute_candidate, apply_approvals,
                        preserve_trace_controls, invalidate_stale_review)


st.set_page_config(page_title='LDHA | Biphasic time courses', layout='wide',
                   page_icon='🧪')
BLUE, ORANGE, GREEN, PURPLE, GRAY = '#0072B2','#D55E00','#009E73','#CC79A7','#4A4A4A'

st.title('LDHA · two-phase time-course analyzer')
st.caption('Exploratory fitting of NADH absorbance time courses at 340 nm')
st.info('This app describes early and late reaction rates. A two-phase trace **does not by itself** '
        'establish formation of a pyruvate–NAD⁺ adduct. Compare independent experiments '
        'and rule out mixing, NADH depletion, and product accumulation.')

with st.sidebar:
    st.header('1 · Upload Cary files')
    uploaded = st.file_uploader('Time-course CSV files', type=['csv','txt'],
                                accept_multiple_files=True,
                                help='Select one or many raw A340 files.')
    st.header('2 · Instrument and conversion')
    path_cm = st.number_input('Cuvette optical path length (cm)', min_value=.01,
                              max_value=10.0, value=1.0, step=.1, format='%.2f')
    extinction = st.number_input('NADH ε340 (mM⁻¹ cm⁻¹)', min_value=.01,
                                 max_value=100.0, value=6.22, step=.01, format='%.2f')
    initial_nadh = st.number_input('Initial NADH (μM)', min_value=.01,
                                   value=75.0, step=5.0)
    blank_slope = st.number_input('Optional blank A340 drift (A/s)',
                                  value=0.0, format='%.7f',
                                  help='Subtract this measured blank slope from all fitted slopes.')
    st.header('3 · CSV parsing')
    with st.expander('Advanced: columns and units'):
        t_col = st.number_input('Time column (1 = first)', min_value=1, value=1)
        a_col = st.number_input('Absorbance column (1 = first)', min_value=1, value=2)
        time_unit = st.selectbox('Cary time units', ['seconds','minutes'], index=0)
    st.header('4 · Fit range')
    st.caption('Mixing cutoff defaults to 15 seconds and can be adjusted separately for each trace below.')

st.markdown('**Models:** two selected linear windows · continuous two-slope breakpoint · '
            'smooth exponential slope transition · single straight-line reference.')

if not uploaded:
    st.subheader('Start with a few traces')
    st.markdown('Upload several Cary CSV files. The app tries to read [pyruvate] from filenames '
                'such as `RS_WT_53,3mM_A.csv`; you can edit concentrations in the controls. '
                'For a first check, use low- and high-pyruvate traces from the same enzyme.')
    st.markdown('The downloaded project contains **10 synthetic example CSVs** in `sample_data/` '
                'for testing the interface. These are not experimental measurements.')
    st.stop()

if len({file.name for file in uploaded}) != len(uploaded):
    st.error('Two uploaded files have the same filename. Rename one before uploading.')
    st.stop()

parsed = {}
problems = []
for file in uploaded:
    blob = file.getvalue()
    uid = sha1(blob + file.name.encode()).hexdigest()[:12]
    try:
        df = read_cary_csv(blob, time_column=int(t_col),
                           absorbance_column=int(a_col), time_unit=time_unit)
        parsed[file.name] = {'df': df, 'uid': uid}
    except (DataError, UnicodeError) as e:
        problems.append(f'{file.name}: {e}')
for issue in problems:
    st.warning(issue)
if not parsed:
    st.error('No valid time-course files could be read.')
    st.stop()

# Streamlit otherwise clears a trace's widgets when the user switches files.
# Preserve per-trace fitting windows, selected method, decision, and notes.
preserve_trace_controls(st.session_state, [d['uid'] for d in parsed.values()])

st.subheader('Select experiments and enter concentrations')
st.success(f'{len(parsed)} CSV file(s) successfully read.')
st.caption('Use the controls below to select traces and check concentrations. '
           'These replace the editable table used in the previous version.')
rows = []
for name, data in parsed.items():
    uid = data['uid']
    detected = filename_pyruvate(name)
    with st.container(border=True):
        st.markdown(f'**{name}**')
        include_col, concentration_col, enzyme_col = st.columns([1, 1.5, 1.5])
        with include_col:
            include = st.checkbox('Include', value=True, key=f'include_{uid}')
        with concentration_col:
            concentration = st.number_input(
                '[Pyruvate] (mM)', min_value=0.0,
                value=float(detected) if detected is not None else 0.0,
                step=0.01, format='%.4f', key=f'concentration_{uid}',
                help='Check this against the actual experimental concentration.')
        with enzyme_col:
            enzyme = st.text_input('Enzyme / variant', value='WT',
                                   key=f'enzyme_{uid}')
        if detected is None:
            st.caption('Concentration was not found in the filename. Enter it manually above.')
        rows.append({'Filename': name, '[Pyruvate] (mM)': concentration,
                     'Enzyme': enzyme, 'Include': include})
metadata = pd.DataFrame(rows)
selected_rows = metadata[metadata['Include'].fillna(False)].copy()
if selected_rows.empty:
    st.warning('Select at least one trace to fit.')
    st.stop()

st.subheader('Choose the early and late linear regions')
names = selected_rows['Filename'].tolist()
current = st.selectbox('Inspect and edit windows for', names,
                       help='These window choices are saved in your browser session only.')

for name, obj in parsed.items():
    prefix = obj['uid']
    tm = obj['df'].time_s.to_numpy(float)
    lo, hi = float(tm.min()), float(tm.max())
    dt = float(np.median(np.diff(np.unique(tm))))
    cutoff_key = f'{prefix}_mixing_cutoff'
    if cutoff_key not in st.session_state:
        st.session_state[cutoff_key] = float(min(15., max(lo, hi-15*dt)))
    try:
        defaults = recommended_windows(tm, cutoff_s=st.session_state[cutoff_key])
    except DataError:
        defaults = {'early_start': lo, 'early_end': hi,
                    'late_start': lo, 'late_end': hi}
    for part, value in defaults.items():
        key = f'{prefix}_{part}'
        if key not in st.session_state:
            st.session_state[key] = round(float(value), 3)
    for part, value in [('single_start', defaults['early_start']),
                        ('single_end', defaults['early_end'])]:
        key = f'{prefix}_{part}'
        if key not in st.session_state:
            st.session_state[key] = round(float(value), 3)

curr_uid = parsed[current]['uid']; curr_df = parsed[current]['df']
tmin, tmax = float(curr_df.time_s.min()), float(curr_df.time_s.max())
st.number_input('Exclude mixing artifact before this time (s) — for this trace',
                min_value=0.0, max_value=tmax, step=1.0,
                key=f'{curr_uid}_mixing_cutoff',
                help='Default 15 s. The mixing period is excluded from all fits for this trace. '
                     'You may choose a different cutoff for each file.')
if float(st.session_state[f'{curr_uid}_early_start']) < float(st.session_state[f'{curr_uid}_mixing_cutoff']):
    st.caption('Early start currently precedes the mixing cutoff; move it later to use this trace.')
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.number_input('Early start (s)', min_value=tmin, max_value=tmax,
                    step=0.5, key=f'{curr_uid}_early_start')
with c2:
    st.number_input('Early end (s)', min_value=tmin, max_value=tmax,
                    step=0.5, key=f'{curr_uid}_early_end')
with c3:
    st.number_input('Late start (s)', min_value=tmin, max_value=tmax,
                    step=0.5, key=f'{curr_uid}_late_start')
with c4:
    st.number_input('Late end (s)', min_value=tmin, max_value=tmax,
                    step=0.5, key=f'{curr_uid}_late_end')

if st.button('Copy mixing cutoff and fitting windows to all included traces', help='Useful for runs sampled over the same time interval'):
    for name in names:
        if name == current:
            continue
        dest = parsed[name]
        lo, hi = float(dest['df'].time_s.min()), float(dest['df'].time_s.max())
        for part in ('mixing_cutoff','early_start','early_end','late_start','late_end'):
            st.session_state[f"{dest['uid']}_{part}"] = float(np.clip(
                st.session_state[f'{curr_uid}_{part}'], lo, hi))
    st.success('Cutoff and window limits copied. Inspect each trace before interpreting results.')

st.caption('All fits exclude the selected mixing period (default 15 s). '
           'The two linear fits use their selected windows. The segmented and exponential models '
           'use the remaining time course. Later-phase fits are intentionally *not* restricted '
           'to the initial-rate (<10% conversion) regime.')

results = []
fits = {}
for _, row in selected_rows.iterrows():
    name = row['Filename']; d = parsed[name]
    t = d['df'].time_s.to_numpy(float); a = d['df'].absorbance.to_numpy(float)
    uid = d['uid']; label = str(row['Enzyme']).strip() or 'Unspecified'
    try:
        s = float(row['[Pyruvate] (mM)'])
    except (ValueError, TypeError):
        s = np.nan
    rec = {'file': name, 'enzyme': label, 'pyruvate_mM': s,
           'n_points': len(t), 'issue': '', 'timecourse_outcome': 'Not assessed',
           'mixing_cutoff_s': float(st.session_state[f'{uid}_mixing_cutoff'])}
    early = late = models = None
    try:
        limits = {k: float(st.session_state[f'{uid}_{k}'])
                  for k in ('early_start','early_end','late_start','late_end')}
        cutoff = float(st.session_state[f'{uid}_mixing_cutoff'])
        if limits['early_start'] < cutoff or limits['late_start'] < cutoff:
            raise DataError(f'A fitting window starts before mixing cutoff ({cutoff:g} s). Adjust the windows.')
        if limits['early_start'] >= limits['early_end'] or limits['late_start'] >= limits['late_end']:
            raise DataError('Time-window start must precede its end.')
        if limits['early_end'] > limits['late_start']:
            raise DataError('Early and late fitting windows overlap. Please separate them.')
        early = fit_line(t, a, limits['early_start'], limits['early_end'])
        late = fit_line(t, a, limits['late_start'], limits['late_end'])
        v1 = signal_slope_to_velocity(early['slope_A_s'], extinction, path_cm, blank_slope)
        v2 = signal_slope_to_velocity(late['slope_A_s'], extinction, path_cm, blank_slope)
        rec.update(v_early_uM_s=v1, v_late_uM_s=v2,
                   early_se_uM_s=1000*early['slope_se_A_s']/(extinction*path_cm),
                   late_se_uM_s=1000*late['slope_se_A_s']/(extinction*path_cm),
                   v_late_over_early=v2/v1 if v1 != 0 else np.nan,
                   early_r2=early['r2'],late_r2=late['r2'],
                   early_n=early['n'],late_n=late['n'],
                   early_start_s=early['start_s'],early_end_s=early['end_s'],
                   late_start_s=late['start_s'],late_end_s=late['end_s'])
        models = fit_all_models(d['df'], dead_time_s=cutoff)
        trans, seg = models['transition'], models['segmented']
        outcome = classify_timecourse(models)
        rec.update(timecourse_outcome=outcome['label'],
                   outcome_reason=outcome['reason'],
                   suggested_next_step=outcome['next_step'],
                   aicc_improvement_exponential=outcome['exponential_aicc_gain'],
                   aicc_improvement_segmented=outcome['segmented_aicc_gain'],
                   exponential_k_bound_side=outcome['k_bound_side'],
                   k_obs_interpretable=outcome['k_interpretable'])
        rec.update(k_obs_s_1=trans['k_obs_s_1'],
                   t_half_s=trans['t_half_s'],
                   v_transition_initial_uM_s=signal_slope_to_velocity(
                       trans['m_initial_A_s'], extinction, path_cm, blank_slope),
                   v_transition_final_uM_s=signal_slope_to_velocity(
                       trans['m_final_A_s'], extinction, path_cm, blank_slope),
                   breakpoint_s=seg['break_s'],
                   v_segmented_initial_uM_s=signal_slope_to_velocity(
                       seg['m_initial_A_s'], extinction, path_cm, blank_slope),
                   v_segmented_final_uM_s=signal_slope_to_velocity(
                       seg['m_final_A_s'], extinction, path_cm, blank_slope),
                   aicc_single=models['linear']['aicc'],
                   aicc_segmented=seg['aicc'],
                   aicc_exponential=trans['aicc'],
                   exponential_at_k_bound=trans['k_at_bound'])
        # Only changes AFTER the mixing cutoff are evaluated, never absolute A340.
        # These are approximate NADH-equivalent changes, not true conversion:
        # pyruvate and products may contribute to time-dependent absorbance.
        postmix_t = t[t >= cutoff]
        rec['apparent_fraction_to_early_end_postmix'] = apparent_postmix_fraction(
            t, a, float(postmix_t[0]), limits['early_end'],
            extinction, path_cm, initial_nadh)
        rec['apparent_fraction_to_late_end_postmix'] = apparent_postmix_fraction(
            t, a, float(postmix_t[0]), limits['late_end'],
            extinction, path_cm, initial_nadh)
        rec['apparent_fraction_full_postmix'] = apparent_postmix_fraction(
            t, a, float(postmix_t[0]), float(postmix_t[-1]),
            extinction, path_cm, initial_nadh)
        quality = []
        if rec['apparent_fraction_to_early_end_postmix'] > .1:
            quality.append('early window >10% apparent NADH-equivalent change after mixing; not an initial-rate window')
        if v1 <= 0 or v2 <= 0:
            quality.append('nonpositive phase velocity')
        if outcome['label'] == 'Transition rate unresolved':
            quality.append('time-dependent transition suggested, but k_obs is unresolved')
        rec['issue'] = '; '.join(quality)
    except (DataError, ValueError, np.linalg.LinAlgError) as exc:
        rec['issue'] = str(exc)
    results.append(rec)
    fits[name] = {'early': early, 'late': late, 'models': models, 'mixing_cutoff_s': float(st.session_state[f'{uid}_mixing_cutoff']), 'limits':
                  {k: st.session_state.get(f'{uid}_{k}') for k in
                   ('early_start','early_end','late_start','late_end')}}

summary = pd.DataFrame(results)
valid = summary[summary['v_early_uM_s'].notna()] if 'v_early_uM_s' in summary else pd.DataFrame()
colA, colB, colC = st.columns(3)
colA.metric('Files selected',len(selected_rows))
colB.metric('Successfully fitted',len(valid))
colC.metric('Needs review', int(summary['issue'].fillna('').astype(str).str.strip().ne('').sum()))
if valid.empty:
    st.error('None of the traces could be fitted. Adjust time windows and inspect the parsing settings.')
    st.dataframe(summary[['file','issue']], hide_index=True)
    st.stop()

st.subheader('Time course and fit comparison')
# Keep the detailed-trace models available to the report exporter in either view.
selected = fits[current]
models = selected['models']
plot_view = st.radio('Display', ('One time course', 'All time courses (grid)'),
                     horizontal=True, key='timecourse_view',
                     help='The grid shows every included trace, sorted by pyruvate concentration.')
if plot_view == 'All time courses (grid)':
    settings_col1, settings_col2, settings_col3 = st.columns([1, 1, 1])
    with settings_col1:
        grid_cols = st.selectbox('Plots per row', [2, 3], index=0)
    with settings_col2:
        grid_smooth = st.checkbox('Show exponential fit', value=False)
    with settings_col3:
        grid_segmented = st.checkbox('Show segmented fit', value=False)

    plot_rows = []
    for _, row in selected_rows.iterrows():
        name = row['Filename']
        concentration = pd.to_numeric(row['[Pyruvate] (mM)'], errors='coerce')
        plot_rows.append((name, float(concentration) if pd.notna(concentration) else float('inf'),
                          str(row['Enzyme']).strip() or 'Unspecified'))
    plot_rows.sort(key=lambda x: (x[1], x[2].lower(), x[0].lower()))
    nrows = int(np.ceil(len(plot_rows) / grid_cols))
    subtitles = []
    for name, conc, enzyme in plot_rows:
        conc_label = f'{conc:.4g} mM' if np.isfinite(conc) else '[S] unknown'
        flagged = summary.loc[summary['file'] == name, 'issue']
        caution = ' · review' if not flagged.empty and str(flagged.iloc[0]).strip() else ''
        subtitles.append(f'{conc_label} · {enzyme}{caution}')
    grid = make_subplots(rows=nrows, cols=grid_cols, subplot_titles=subtitles,
                         vertical_spacing=min(0.09, 0.17 / nrows),
                         horizontal_spacing=0.075 if grid_cols == 2 else 0.055)
    # Use the same time scale to make comparisons between concentrations meaningful.
    global_tmax = max(float(parsed[name]['df'].time_s.max()) for name, _, _ in plot_rows)
    for i, (name, conc, enzyme) in enumerate(plot_rows):
        rr, cc = divmod(i, grid_cols)
        rr += 1
        cc += 1
        df = parsed[name]['df']
        plot_fit = fits[name]
        grid.add_vrect(x0=0, x1=plot_fit['mixing_cutoff_s'], fillcolor='#AAAAAA', opacity=.09, line_width=0, row=rr, col=cc)
        grid.add_trace(go.Scatter(x=df.time_s, y=df.absorbance, mode='lines+markers',
                                  line=dict(color=GRAY, width=1.25), marker=dict(size=2),
                                  name='A340 data', legendgroup='data', showlegend=i == 0,
                                  hovertemplate='t=%{x:.2f} s<br>A340=%{y:.5f}<extra></extra>'),
                       row=rr, col=cc)
        for label, key, color in [('Early linear', 'early', BLUE),
                                  ('Late linear', 'late', ORANGE)]:
            fit = plot_fit[key]
            if fit:
                tx = np.linspace(fit['start_s'], fit['end_s'], 40)
                grid.add_trace(go.Scatter(x=tx,
                                          y=fit['intercept_A'] + fit['slope_A_s'] * tx,
                                          mode='lines', name=label,
                                          legendgroup=key, showlegend=i == 0,
                                          line=dict(color=color, width=3),
                                          hovertemplate='t=%{x:.2f} s<br>A340=%{y:.5f}<extra></extra>'),
                               row=rr, col=cc)
        if plot_fit['models'] is not None:
            m = plot_fit['models']
            for enabled, label, key, color, dash in (
                (grid_smooth, 'Exponential', 'transition', GREEN, 'dash'),
                (grid_segmented, 'Segmented', 'segmented', PURPLE, 'dot')):
                if enabled:
                    grid.add_trace(go.Scatter(x=m['t'], y=m[key]['predicted_A'],
                                              mode='lines', name=label,
                                              legendgroup=key, showlegend=i == 0,
                                              line=dict(color=color, width=1.75, dash=dash)),
                                   row=rr, col=cc)
        grid.update_xaxes(range=[0, global_tmax], row=rr, col=cc,
                          title_text='Time (s)' if rr == nrows else None)
        grid.update_yaxes(title_text='A340' if cc == 1 else None, row=rr, col=cc)
    grid.update_layout(height=max(420, 325*nrows),
                       margin=dict(l=35, r=18, t=80, b=45),
                       legend=dict(orientation='h', yanchor='bottom', y=1.015, x=0),
                       hovermode='closest')
    st.plotly_chart(grid, use_container_width=True)
    st.caption('Each panel has its own vertical absorbance scale, but all panels share a time scale. '
               'The colored lines represent the currently selected fitting windows for that file. '
               '“Review” means the app reported a fitting or data-quality warning. '
               'Select **One time course** for an enlarged plot and model diagnostics.')
    with st.expander('Early and late velocities for all displayed traces'):
        cols = ['file', 'pyruvate_mM', 'v_early_uM_s', 'v_late_uM_s',
                'v_late_over_early', 'timecourse_outcome', 'issue']
        st.dataframe(summary[[c for c in cols if c in summary]],
                     hide_index=True, use_container_width=True)
else:
    selected = fits[current]
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=selected['mixing_cutoff_s'], fillcolor='#AAAAAA', opacity=.1, line_width=0)
    fig.add_trace(go.Scatter(x=curr_df.time_s,y=curr_df.absorbance, mode='lines+markers',
                             marker={'size':4}, line={'color':GRAY, 'width':1},
                             name='Measured A340'))
    for label, fit, color in (('Early linear',selected['early'],BLUE),
                              ('Late linear',selected['late'],ORANGE)):
        if fit:
            tx = np.linspace(fit['start_s'], fit['end_s'], 40)
            fig.add_vrect(x0=fit['start_s'], x1=fit['end_s'],
                          fillcolor=color,opacity=.10,line_width=0)
            fig.add_trace(go.Scatter(x=tx, y=fit['intercept_A']+fit['slope_A_s']*tx,
                                     mode='lines', name=label,
                                     line={'color':color,'width':4}))
    models = selected['models']
    if models:
        for label,key,color,dash in [('Smooth exponential','transition',GREEN,'dash'),
                                      ('Continuous two-line','segmented',PURPLE,'dot')]:
            fig.add_trace(go.Scatter(x=models['t'],y=models[key]['predicted_A'],
                                     mode='lines',name=label,
                                     line={'color':color,'width':2,'dash':dash}))
        fig.add_vline(x=models['segmented']['break_s'], line_dash='dot', line_color=PURPLE)
    fig.update_layout(xaxis_title='Time (s)',yaxis_title='Absorbance (340 nm)',
                      hovermode='x unified',height=485, margin=dict(l=30,r=30,t=25,b=30))
    st.plotly_chart(fig,use_container_width=True)

    row = valid[valid['file']==current]
    if len(row):
        r=row.iloc[0]
        cc1,cc2,cc3,cc4=st.columns(4)
        cc1.metric('Early v (μM/s)',f"{r['v_early_uM_s']:.4g}")
        cc2.metric('Late v (μM/s)',f"{r['v_late_uM_s']:.4g}")
        cc3.metric('Late / early',f"{r['v_late_over_early']:.3f}" if np.isfinite(r['v_late_over_early']) else '—')
        cc4.metric('Exponential k_obs (s⁻¹)',
                   f"{r['k_obs_s_1']:.3g}" if r.get('k_obs_interpretable', False)
                   and pd.notna(r['k_obs_s_1']) else 'Not resolved')
        st.info(f"**{r['timecourse_outcome']}** — {r['outcome_reason']}\n\n"
                f"**Suggested action:** {r['suggested_next_step']}")
        if str(r['issue']):
            st.warning(str(r['issue']))
    else:
        st.warning('This trace did not fit with the selected windows. See the issues table below.')

    with st.expander('Fit diagnostics for the selected trace'):
        if models:
            diagnostic = pd.DataFrame([
                {'Model':'Single straight line','Parameters':2,**models['linear']},
                {'Model':'Continuous two-line breakpoint','Parameters':4,**models['segmented']},
                {'Model':'Smooth exponential slope transition','Parameters':4,**models['transition']}
            ]).drop(columns=['predicted_A'],errors='ignore')
            st.dataframe(diagnostic[['Model','Parameters','n','r2','rmse_A','aicc']],
                         hide_index=True, use_container_width=True)
            st.caption('AICc is comparable only because all three models fit the same retained '
                       'time points. The best (lowest) AICc is not proof of a particular chemical mechanism. '
                       'Breakpoint-search AICc is approximate.')

# One decision per trace. Model selection is available as an override for the
# currently inspected trace; no separate checkbox or per-trace approval button.
st.subheader('Choose the analysis to use')
st.caption('Each trace needs only one decision: kinetic analysis, descriptive only, '
           'or exclude. The automatic recommendations are conservative starting points. '
           'The bulk action applies them for the whole experiment; check the choices below. '
           'No decisions are stored online: download the results before closing the browser.')


def analysis_signature(name):
    """Hash all scientific inputs; do NOT include the user's decision or note."""
    d = parsed[name]; uid = d['uid']
    sr = summary.loc[summary['file'] == name].iloc[0]
    method_value = st.session_state.get(f'approval_method_{uid}',
                                        suggest_method(str(sr['timecourse_outcome'])))
    row_meta = selected_rows.loc[selected_rows['Filename'] == name].iloc[0]
    vals = dict(file_uid=uid, pyruvate_mM=float(row_meta['[Pyruvate] (mM)']),
                enzyme=str(row_meta['Enzyme']), method=method_value,
                extinction=float(extinction), path_cm=float(path_cm),
                blank_slope=float(blank_slope), initial_nadh=float(initial_nadh))
    vals.update({x: float(st.session_state[f'{uid}_{x}']) for x in
                 ('mixing_cutoff', 'early_start', 'early_end', 'late_start',
                  'late_end', 'single_start', 'single_end')})
    return analysis_fingerprint(vals)


def candidate_for(name, method_override=None):
    d = parsed[name]; uid = d['uid']
    sr = summary.loc[summary['file'] == name].iloc[0]
    method_name = method_override or st.session_state.get(
        f'approval_method_{uid}', suggest_method(str(sr['timecourse_outcome'])))
    try:
        return compute_candidate(
            method_name, d['df'].time_s.to_numpy(float),
            d['df'].absorbance.to_numpy(float), fits[name],
            single_start_s=float(st.session_state[f'{uid}_single_start']),
            single_end_s=float(st.session_state[f'{uid}_single_end']),
            cutoff_s=float(st.session_state[f'{uid}_mixing_cutoff']),
            extinction=extinction, path_cm=path_cm, blank_slope=blank_slope,
            initial_nadh=initial_nadh, outcome=str(sr['timecourse_outcome']))
    except (DataError, ValueError, TypeError, KeyError):
        return None


signatures = st.session_state.setdefault('_review_input_signatures', {})

# The bulk action deliberately chooses DESCRIPTIVE for time-dependent and
# unresolved traces. It never automatically assumes their postmix model v0
# is a trustworthy initial velocity.
if st.button('Accept recommended analyses for all traces', type='primary'):
    counts = {KINETIC: 0, DESCRIPTIVE: 0, EXCLUDED: 0}
    for name in names:
        uid = parsed[name]['uid']
        sr = summary.loc[summary['file'] == name].iloc[0]
        st.session_state[f'approval_method_{uid}'] = suggest_method(str(sr['timecourse_outcome']))
        cand = candidate_for(name)
        decision = recommended_decision(str(sr['timecourse_outcome']), cand,
                                        str(sr.get('issue', '')))
        st.session_state[f'approval_decision_{uid}'] = decision
        signatures[name] = analysis_signature(name)
        counts[decision] += 1
    st.success(f"Recommendations applied: {counts[KINETIC]} kinetic, "
               f"{counts[DESCRIPTIVE]} descriptive only, {counts[EXCLUDED]} excluded. "
               'Review all choices before exporting the primary-rate fit.')

current_row = summary.loc[summary['file'] == current].iloc[0]
current_fit = fits[current]
current_outcome = str(current_row['timecourse_outcome'])
suggested = suggest_method(current_outcome)
method_key = f'approval_method_{curr_uid}'
if method_key not in st.session_state:
    st.session_state[method_key] = suggested
with st.expander(f'Inspect or change the fitted method for {current}', expanded=False):
    st.markdown(f'Screening outcome: **{current_outcome}**  \n'
                f'Suggested method: **{suggested}**')
    method = st.selectbox('Method for this trace', METHODS, key=method_key,
                          help='The chosen method applies to this trace only. '
                               'After changing the method, review the acceptance decision again.')
    if method != suggested:
        st.caption('Method overridden. Explain the choice in the researcher note if useful.')
    if method == SINGLE:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.number_input('Single linear start (s)', min_value=tmin, max_value=tmax,
                            step=.5, key=f'{curr_uid}_single_start')
        with sc2:
            st.number_input('Single linear end (s)', min_value=tmin, max_value=tmax,
                            step=.5, key=f'{curr_uid}_single_end')

    candidate_current = candidate_for(current)
    if candidate_current is None:
        st.warning('The chosen fit is unavailable. Correct the mixing cutoff or fitting windows.')
    else:
        if candidate_current['v0_uM_s'] is not None:
            st.write(f"**Candidate velocity:** {candidate_current['v0_uM_s']:.5g} µM/s"
                     f" · {candidate_current['basis']}")
        if candidate_current['k_obs_s_1'] is not None:
            st.write(f"**Exponential k_obs:** {candidate_current['k_obs_s_1']:.4g} s⁻¹")
        if candidate_current.get('warning'):
            st.warning(candidate_current['warning'])
        if method != EXCLUDE and current_fit['models'] is not None:
            with st.expander('Fit residuals', expanded=False):
                tt = curr_df.time_s.to_numpy(float)
                yy = curr_df.absorbance.to_numpy(float)
                if method in (SINGLE, EARLY):
                    line = (candidate_current['single_fit'] if method == SINGLE
                            else current_fit['early'])
                    mask = (tt >= line['start_s']) & (tt <= line['end_s'])
                    fit_t = tt[mask]
                    predicted = line['intercept_A'] + line['slope_A_s'] * fit_t
                else:
                    fit_t = current_fit['models']['t']
                    mask = tt >= float(st.session_state[f'{curr_uid}_mixing_cutoff'])
                    predicted = current_fit['models']['transition']['predicted_A']
                residual_fig = go.Figure()
                residual_fig.add_trace(go.Scatter(x=fit_t, y=yy[mask]-predicted,
                                                   mode='markers', marker=dict(color=BLUE, size=5)))
                residual_fig.add_hline(y=0, line_dash='dash', line_color=GRAY)
                residual_fig.update_layout(height=230, xaxis_title='Time (s)',
                                           yaxis_title='A340 residual',
                                           margin=dict(l=25,r=15,t=15,b=25))
                st.plotly_chart(residual_fig, use_container_width=True)

note_key = f'approval_note_{curr_uid}'
with st.expander('Optional researcher note for selected trace', expanded=False):
    st.text_input('Reason for override, exclusion, or uncertainty', key=note_key)

# A scientific setting change cancels the old selection. A researcher can
# reselect one decision (no second confirmation button is necessary).
for name in names:
    invalidate_stale_review(st.session_state, signatures, name,
                            parsed[name]['uid'], analysis_signature(name))

st.markdown('**Review one decision per time course**')
st.caption('Kinetic analysis places the selected velocity into the primary v₀ plot. '
           'Descriptive only retains the method and velocity without using it in the primary fit. '
           'Exclude omits the trace from the accepted-rate analyses. '
           'Changing a fitting window or scientific input resets that trace to Not reviewed.')

approvals = {}
review_summary_rows = []
for name in names:
    uid = parsed[name]['uid']
    sr = summary.loc[summary['file'] == name].iloc[0]
    method = st.session_state.get(f'approval_method_{uid}',
                                  suggest_method(str(sr['timecourse_outcome'])))
    cand = candidate_for(name)
    choices = available_decisions(cand)
    dec_key = f'approval_decision_{uid}'
    if dec_key not in st.session_state or st.session_state[dec_key] not in choices:
        st.session_state[dec_key] = NOT_REVIEWED
    conc = float(sr['pyruvate_mM'])
    with st.container(border=True):
        left, right = st.columns([2.2, 2.0])
        with left:
            st.markdown(f'**{name}** · {conc:.4g} mM')
            st.caption(f"{sr['timecourse_outcome']} · Method: {method}"
                       + (' · Fit needs review' if str(sr.get('issue', '')).strip() else ''))
        with right:
            chosen = st.selectbox('Analysis decision', choices, key=dec_key,
                                  help='One selection is sufficient; no Accept button is needed.')
    if chosen != NOT_REVIEWED:
        accepted_candidate = (candidate_for(name, EXCLUDE) if chosen == EXCLUDED else cand)
        if accepted_candidate is not None:
            approvals[name] = dict(
                fingerprint=signatures[name], candidate=accepted_candidate,
                primary_verified=(chosen == KINETIC),
                decision=chosen,
                note=str(st.session_state.get(f'approval_note_{uid}', '')))
    review_summary_rows.append(dict(file=name, pyruvate_mM=conc,
                                    classification=sr['timecourse_outcome'],
                                    method=method, decision=chosen,
                                    velocity_uM_s=(cand or {}).get('v0_uM_s')))

fingerprints = {name: analysis_signature(name) for name in names}
summary_accepted = apply_approvals(summary, approvals, fingerprints)
review_counts = pd.Series([r['decision'] for r in review_summary_rows]).value_counts()
st.caption(f"Decisions: {int(review_counts.get(KINETIC, 0))} kinetic · "
           f"{int(review_counts.get(DESCRIPTIVE, 0))} descriptive · "
           f"{int(review_counts.get(EXCLUDED, 0))} excluded · "
           f"{int(review_counts.get(NOT_REVIEWED, 0))} not reviewed")
with st.expander('Compact decision summary', expanded=False):
    st.dataframe(pd.DataFrame(review_summary_rows), hide_index=True,
                 use_container_width=True)

primary = summary_accepted.loc[summary_accepted['primary_verified'] == True].copy()
primary = primary[(pd.to_numeric(primary['accepted_v0_uM_s'], errors='coerce') > 0) &
                  (pd.to_numeric(primary['pyruvate_mM'], errors='coerce') > 0)]
st.subheader('Primary initial-velocity dataset — selected kinetic analyses')
st.caption('This dataset contains velocities selected for primary kinetic analysis. '
           'Exponential starting velocities refer to the first retained time point after mixing, '
           'not t = 0. Manual approval is not proof of an initial-rate condition; '
           'the exported method and warning fields preserve that distinction.')
if primary.empty:
    st.info('No velocities have been selected for kinetic analysis yet. Choose Kinetic analysis '
            'for suitable traces or apply the recommended decisions and review them.')
else:
    shown = ['file', 'enzyme', 'pyruvate_mM', 'accepted_method',
             'accepted_v0_uM_s','accepted_v0_se_uM_s',
             'accepted_reference_time_s','accepted_k_obs_s_1','accepted_caution']
    st.dataframe(primary[shown], hide_index=True, use_container_width=True)

primary_fits = {}
if not primary.empty:
    selected_enzymes = sorted(primary.enzyme.unique())
    primary_enzyme = (st.selectbox('Primary curve: enzyme/variant', selected_enzymes,
                                   key='primary_enzyme') if len(selected_enzymes) > 1
                      else selected_enzymes[0])
    # Fit every enzyme with sufficient accepted velocities for the exported PDF,
    # even when only one enzyme is displayed on screen.
    for enzyme_name, enzyme_rows in primary.groupby('enzyme'):
        try:
            primary_fits[enzyme_name] = fit_substrate_inhibition(
                enzyme_rows.pyruvate_mM.to_numpy(float),
                enzyme_rows.accepted_v0_uM_s.to_numpy(float))
        except DataError:
            pass
    primary_group = primary[primary.enzyme == primary_enzyme]
    primary_fig = go.Figure()
    for method_label, points in primary_group.groupby('accepted_method'):
        primary_fig.add_trace(go.Scatter(
            x=points.pyruvate_mM, y=points.accepted_v0_uM_s,
            mode='markers', name=method_label,
            marker=dict(size=10), text=points.file,
            hovertemplate='%{text}<br>[Pyr]=%{x:.4g} mM<br>v₀=%{y:.4g} μM/s<extra></extra>'))
    if primary_enzyme in primary_fits:
        ft = primary_fits[primary_enzyme]
        x = np.linspace(0, float(primary_group.pyruvate_mM.max()), 400)
        primary_fig.add_trace(go.Scatter(x=x,
                           y=substrate_inhibition(x, ft['vmax_uM_s'], ft['km_mM'], ft['ki_mM']),
                           name='Accepted-v₀ substrate-inhibition fit',mode='lines',
                           line=dict(color=BLUE, width=2)))
        st.markdown(f"**Descriptive accepted-v₀ fit:** Vmax = {ft['vmax_uM_s']:.4g} μM/s; "
                    f"KM = {ft['km_mM']:.4g} mM; KI = {ft['ki_mM']:.4g} mM. "
                    f"{ft.get('warning', '')}")
    else:
        st.caption('At least five distinct positive concentrations selected for kinetic analysis '
                   'are required for the primary substrate-inhibition fit.')
    primary_fig.update_layout(xaxis_title='[Pyruvate] (mM)', yaxis_title='Accepted v₀ (μM/s)',
                              height=390,margin=dict(l=30,r=25,t=15,b=30))
    st.plotly_chart(primary_fig, use_container_width=True)

st.subheader('Velocities versus [pyruvate]')
if summary['enzyme'].nunique() > 1:
    enzymes = sorted(summary['enzyme'].unique())
    target_enzyme = st.selectbox('Show enzyme/variant', enzymes)
else:
    target_enzyme = summary['enzyme'].iloc[0]
trace_group = valid[valid['enzyme']==target_enzyme].copy()
trace_group = trace_group[pd.to_numeric(trace_group['pyruvate_mM'],errors='coerce').notna()]

substrate_fits = {}
kinetics_fig = go.Figure()
for phase,col,color,label in [('v_early_uM_s','early_se_uM_s',BLUE,'Early-window v'),
                               ('v_late_uM_s','late_se_uM_s',ORANGE,'Late-window v')]:
    if len(trace_group):
        err = trace_group[col].fillna(0).to_numpy(float) if col in trace_group else None
        kinetics_fig.add_trace(go.Scatter(x=trace_group['pyruvate_mM'],
                        y=trace_group[phase],mode='markers',name=label,
                        marker={'color':color,'size':9},error_y=dict(type='data',array=err,visible=True),
                        text=trace_group['file'],hovertemplate='%{text}<br>S=%{x:.4g} mM<br>v=%{y:.4g} μM/s<extra></extra>'))
        try:
            fit=fit_substrate_inhibition(trace_group.pyruvate_mM.to_numpy(float),
                                        trace_group[phase].to_numpy(float))
            substrate_fits[phase]=fit
            smax=float(trace_group.pyruvate_mM.max())
            curve=np.linspace(0,smax,350)
            kinetics_fig.add_trace(go.Scatter(x=curve,
                      y=substrate_inhibition(curve,fit['vmax_uM_s'],fit['km_mM'],fit['ki_mM']),
                      name=f'{label} · inhibition fit',mode='lines',
                      line=dict(color=color, width=2)))
        except DataError:
            pass
kinetics_fig.update_layout(xaxis_title='[Pyruvate] (mM)',
                           yaxis_title='NADH consumption rate (μM/s)',height=410,
                           margin=dict(l=30,r=25,t=20,b=30),hovermode='closest')
st.plotly_chart(kinetics_fig,use_container_width=True)

ratio_fig=go.Figure()
if len(trace_group):
    ratio_fig.add_trace(go.Scatter(x=trace_group.pyruvate_mM,
                           y=trace_group.v_late_over_early,mode='markers+lines',
                           marker=dict(color=PURPLE,size=9),name='Late / early'))
ratio_fig.add_hline(y=1, line_dash='dash',line_color=GRAY)
ratio_fig.update_layout(xaxis_title='[Pyruvate] (mM)',yaxis_title='Late velocity / early velocity',
                        height=280,margin=dict(l=30,r=25,t=15,b=30))
st.plotly_chart(ratio_fig,use_container_width=True)

if substrate_fits:
    st.markdown('**Substrate-inhibition fits (separate apparent parameter sets)**')
    rows=[]
    for phase, fit in substrate_fits.items():
        rows.append({'Phase':phase.replace('v_','').replace('_uM_s',''),
                     'Vmax (μM/s)':fit['vmax_uM_s'],
                     'KM (mM)':fit['km_mM'],
                     'KI (mM)':fit['ki_mM'],
                     'R²':fit['r2'], 'Caution':fit['warning']})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.caption('Model: v = Vmax[S] / (KM + [S] + [S]²/KI). '
               'Because time-dependent phases need not be steady state, Vmax, KM, and KI '
               'here are descriptive *apparent* parameters, not necessarily microscopic constants.')
else:
    st.caption('To fit early- and late-phase substrate-inhibition curves, provide at least '
               'five different positive pyruvate concentrations with valid positive velocities.')

st.subheader('Results and downloads')
st.caption('Apparent fraction columns describe A340 changes after the per-trace mixing cutoff, '
           'expressed in NADH equivalents. They do not include consumption before the cutoff '
           'and do not correct for changes in pyruvate or product absorbance. '
           'A late phase beyond 10% is not automatically a failed fit. '
           'High baseline absorbance at concentrated pyruvate warrants substrate-matched blanks, '
           'but it is not automatically a failed measurement.')
st.dataframe(summary_accepted,hide_index=True,use_container_width=True)
st.caption('The time-course outcomes are exploratory screening labels (AICc improvement ≥10 '
           'and modeled slope change ≥10%), not formal significance tests or evidence of '
           'pyruvate–NAD⁺ adduct formation. They do not account for serially correlated noise.')

csv_content=summary_accepted.to_csv(index=False).encode('utf-8-sig')
col1,col2,col3=st.columns(3)
with col1:
    st.download_button('Download full velocity summary (CSV)',csv_content,
                       file_name='LDHA_two_phase_velocities.csv',mime='text/csv')
with col2:
    if models and selected['early'] and selected['late']:
        settings={'Selected trace':current,
                  'Extinction coefficient (mM^-1 cm^-1)':extinction,
                  'Pathlength (cm)':path_cm,
                  'Initial NADH (uM)':initial_nadh,
                  'Blank slope (A/s)':blank_slope,
                  'Mixing cutoff for selected trace (s)': selected['mixing_cutoff_s'],
                  'Accepted status': current_approval['acceptance_status'],
                  'Accepted method': current_approval['accepted_method'],
                  'Selected for kinetic analysis': current_approval['primary_verified'],
                  'Researcher decision': current_approval.get('review_decision', ''),
                  'Accepted v0 (uM/s)': current_approval['accepted_v0_uM_s'],
                  'Postmix apparent fraction to late end (NOT true NADH depletion)':
                      float(summary.loc[summary['file']==current, 'apparent_fraction_to_late_end_postmix'].iloc[0])}
        pdf=make_pdf(current,curr_df,models,selected['early'],selected['late'],
                     trace_group,substrate_fits,settings)
        st.download_button('Download 3-page PDF report',pdf,
                           file_name='LDHA_two_phase_report.pdf',mime='application/pdf')
with col3:
    model_rows=[]
    for name,fit in fits.items():
        if fit['models']:
            m=fit['models']
            for i,time_s in enumerate(m['t']):
                model_rows.append({'file':name,'time_s':time_s,'observed_A':m['A'][i],
                                  'single_A':m['linear']['predicted_A'][i],
                                  'segmented_A':m['segmented']['predicted_A'][i],
                                  'exponential_A':m['transition']['predicted_A'][i]})
    if model_rows:
        st.download_button('Download fitted time courses (CSV)',
                           pd.DataFrame(model_rows).to_csv(index=False).encode('utf-8-sig'),
                           file_name='LDHA_fitted_traces.csv',mime='text/csv')


# Export a true PDF grid instead of relying on browser Print, which may clip
# interactive Plotly rows. This includes all selected traces irrespective of
# which on-screen view is active.
st.markdown('**All time-course plots (PDF)**')
pdf_extra1, pdf_extra2 = st.columns(2)
with pdf_extra1:
    pdf_show_exponential = st.checkbox('Include exponential fits in PDF', value=False)
with pdf_extra2:
    pdf_show_segmented = st.checkbox('Include segmented fits in PDF', value=False)
all_traces_pdf = make_all_traces_pdf(
    parsed, fits, summary_accepted,
    settings={'Mixing cutoff': 'per trace, listed in plots'},
    show_exponential=pdf_show_exponential,
    show_segmented=pdf_show_segmented,
)
st.download_button('Download ALL time courses (PDF)', all_traces_pdf,
                   file_name='LDHA_all_time_courses.pdf', mime='application/pdf',
                   help='Printable landscape PDF: four time courses per page, in increasing pyruvate order.')
st.caption('Includes every selected trace, with its early and late fitted lines. '
           'Plots share a time axis but use separate absorbance scales. '
           'The existing 3-page PDF report remains available above.')

st.markdown('**Download accepted analysis results**')
ac1, ac2 = st.columns(2)
with ac1:
    st.download_button('Download accepted primary v₀ dataset (CSV)',
                       primary.to_csv(index=False).encode('utf-8-sig'),
                       file_name='LDHA_accepted_initial_velocities.csv', mime='text/csv',
                       disabled=primary.empty,
                       help='Only velocities selected for primary kinetic analysis; review experimental initial-rate suitability.')
with ac2:
    accepted_pdf = make_acceptance_pdf(summary_accepted, primary_fits,
                                      settings={'Initial NADH (uM)':initial_nadh,
                                                'Extinction coefficient (mM^-1 cm^-1)':extinction,
                                                'Pathlength (cm)':path_cm,
                                                'Blank slope (A/s)':blank_slope})
    st.download_button('Download accepted-analysis report (PDF)', accepted_pdf,
                       file_name='LDHA_accepted_analysis.pdf', mime='application/pdf',
                       help='All files, selected analysis methods, review notes and primary v0 fits.')

with st.expander('Interpretation notes — what to do next', expanded=True):
    st.markdown('''
**1. No detectable time-dependent change**

- **Interpretation:** The post-mixing time course is adequately described by a straight line under the current experimental conditions. At low [pyruvate], this may be exactly what we expect. It does **not** mean substrate inhibition is absent; a constant but inhibited velocity can be linear.
- **Analysis in this app:** The default method is **Single linear window**. Select **Accept for kinetic analysis** if the window is suitable for an initial rate, or **Descriptive only** if it is not. Do not interpret exponential k_obs.

**2. Detectable time-dependent change**

- **Interpretation:** The exponential model captures appreciable curvature beyond a straight line, and its k_obs is not at a bound. The rate may slow **or** accelerate; curvature alone does not demonstrate a pyruvate–NAD⁺ adduct.
- **Analysis in this app:** Review the **Exponential transition** fit and residuals. The bulk recommendation is **Descriptive only** because the model starting velocity is at the first post-mixing point, **not** necessarily reaction time zero. Select **Accept for kinetic analysis** only if you can justify its use as an initial velocity. Report k_obs only when resolved.

**3. Transition rate unresolved**

- **Interpretation:** The trace suggests curvature, but the exponential k_obs is at a bound or its form is not well supported by the data. **Do not report k_obs as a measured transition rate.**
- **Analysis in this app:** The default is a provisional **Early window** and the bulk recommendation is **Descriptive only**. You may override the method or exclude the trace. Do not accept k_obs. Check mixing and residuals first. If the transition is potentially **faster** than the mixing cutoff, use faster mixing/acquisition; if potentially **slower** than the observation period, collect longer traces with appropriate depletion controls. Repeat under matched conditions. The boundary direction, when available, is listed in the results table.

These three automated outcomes use **exploratory thresholds**: ≥10 AICc units of improvement over a straight line **and** ≥10% modeled change in slope. They are not formal hypothesis tests; closely spaced measurements may have correlated noise. A visible change in slope is not proof of the inhibition mechanism.

**Additional limitations and controls**

- Early/late slopes depend on the selected windows. A visible curve may not have two genuinely linear phases.
- An exponential slope transition is phenomenological. Its **k_obs** is not automatically an adduct formation constant.
- As NADH decreases and NAD⁺ builds up, the reaction environment changes; late-phase rates may be product- or substrate-dependent.
- Mixing artifacts and the instrumental dead time can bias early slopes. Compare mixing-only controls and replicate traces.
- The ~10% conversion rule is a guide for **initial-rate** measurements. It must not be used to discard the later time-course data needed to investigate time-dependent inhibition.
- Displayed 'apparent NADH-equivalent changes' use post-mixing absorbance differences, not absolute A340. They are **not reliable measurements of NADH conversion** unless contributions from pyruvate and products are established.
- High pyruvate concentration can increase the baseline A340 even if its molar absorptivity is much lower than NADH's. Check matched pyruvate blanks and the spectrophotometer's linear absorbance range. Changes in pyruvate absorbance during turnover can also influence the observed slope.
- The software does not perform kinetic modeling of the complete ordered Bi–Bi reaction network or of covalent adduct formation.
- When possible, repeat runs with varied initial NADH and independent mixing/control experiments.
- Data are processed in the active Streamlit session and are not written to an app database. The user downloads results explicitly.
''')
