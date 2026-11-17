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
    DataError, filename_pyruvate, fit_all_models, fit_line,
    fit_substrate_inhibition, read_cary_csv, recommended_windows,
    signal_slope_to_velocity, substrate_inhibition,
)
from report import make_pdf, make_all_traces_pdf


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
    dead_time_s = st.number_input('Exclude data before this time (s)',
                                  min_value=0.0, value=5.0, step=1.0)

st.markdown('**Models:** two selected linear windows · continuous two-slope breakpoint · '
            'smooth exponential slope transition · single straight-line reference.')

if not uploaded:
    st.subheader('Start with a few traces')
    st.markdown('Upload several Cary CSV files. The app tries to read [pyruvate] from filenames '
                'such as `RS_WT_53,3mM_A.csv`; you can edit concentrations in the table. '
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
    defaults = recommended_windows(obj['df'].time_s.to_numpy())
    for part, value in defaults.items():
        key = f'{prefix}_{part}'
        if key not in st.session_state:
            st.session_state[key] = round(float(value), 3)

curr_uid = parsed[current]['uid']; curr_df = parsed[current]['df']
tmin, tmax = float(curr_df.time_s.min()), float(curr_df.time_s.max())
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

if st.button('Copy these windows to all included traces', help='Useful for runs sampled over the same time interval'):
    for name in names:
        if name == current:
            continue
        dest = parsed[name]
        lo, hi = float(dest['df'].time_s.min()), float(dest['df'].time_s.max())
        for part in ('early_start','early_end','late_start','late_end'):
            st.session_state[f"{dest['uid']}_{part}"] = float(np.clip(
                st.session_state[f'{curr_uid}_{part}'], lo, hi))
    st.success('Window limits copied. Inspect other traces before interpreting results.')

st.caption('The two linear fits use only the points inside each selected window. '
           'The segmented and exponential models use all retained points after the mixing/dead-time cutoff.')

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
           'n_points': len(t), 'issue': ''}
    early = late = models = None
    try:
        limits = {k: float(st.session_state[f'{uid}_{k}'])
                  for k in ('early_start','early_end','late_start','late_end')}
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
        models = fit_all_models(d['df'], dead_time_s=dead_time_s)
        trans, seg = models['transition'], models['segmented']
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
        rec['NADH_consumed_fraction_approx'] = max(0, a.max()-a.min())*1000/(extinction*path_cm*initial_nadh)
        quality = []
        if rec['NADH_consumed_fraction_approx'] > .1:
            quality.append('>10% of initial NADH absorbance change')
        if v1 <= 0 or v2 <= 0:
            quality.append('nonpositive phase velocity')
        if trans['k_at_bound']:
            quality.append('exponential k at fitting bound')
        rec['issue'] = '; '.join(quality)
    except (DataError, ValueError, np.linalg.LinAlgError) as exc:
        rec['issue'] = str(exc)
    results.append(rec)
    fits[name] = {'early': early, 'late': late, 'models': models, 'limits':
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
                'v_late_over_early', 'issue']
        st.dataframe(summary[[c for c in cols if c in summary]],
                     hide_index=True, use_container_width=True)
else:
    selected = fits[current]
    fig = go.Figure()
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
        cc4.metric('Exponential k_obs (s⁻¹)',f"{r['k_obs_s_1']:.3g}" if pd.notna(r['k_obs_s_1']) else '—')
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
st.dataframe(summary,hide_index=True,use_container_width=True)

csv_content=summary.to_csv(index=False).encode('utf-8-sig')
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
                  'Dead time excluded (s)':dead_time_s}
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
    parsed, fits, summary,
    settings={'Dead time excluded (s)': dead_time_s},
    show_exponential=pdf_show_exponential,
    show_segmented=pdf_show_segmented,
)
st.download_button('Download ALL time courses (PDF)', all_traces_pdf,
                   file_name='LDHA_all_time_courses.pdf', mime='application/pdf',
                   help='Printable landscape PDF: four time courses per page, in increasing pyruvate order.')
st.caption('Includes every selected trace, with its early and late fitted lines. '
           'Plots share a time axis but use separate absorbance scales. '
           'The existing 3-page PDF report remains available above.')

with st.expander('Scientific limitations and suggested controls'):
    st.markdown('''
- Early/late slopes depend on the selected windows. A visible curve may not have two genuinely linear phases.
- An exponential slope transition is phenomenological. Its **k_obs** is not automatically an adduct formation constant.
- As NADH decreases and NAD⁺ builds up, the reaction environment changes; late-phase rates may be product- or substrate-dependent.
- Mixing artifacts and the instrumental dead time can bias early slopes. Compare mixing-only controls and replicate traces.
- Check the fraction of NADH consumed. If substantial, **neither late nor early slopes necessarily represent initial rates**.
- The software does not perform kinetic modeling of the complete ordered Bi–Bi reaction network or of covalent adduct formation.
- When possible, repeat runs with varied initial NADH and independent mixing/control experiments.
- Data are processed in the active Streamlit session and are not written to an app database. The user downloads results explicitly.
''')
