"""Portable PDF report for the LDHA biphasic app, including no raw data storage."""
from __future__ import annotations
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis import substrate_inhibition, transition_signal



def _primary_rows(accepted_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Only explicitly selected primary velocities; ignore unreviewed data."""
    if accepted_summary is None or accepted_summary.empty:
        return pd.DataFrame()
    needed = {'primary_verified', 'accepted_v0_uM_s', 'pyruvate_mM', 'enzyme'}
    if not needed.issubset(accepted_summary.columns):
        return pd.DataFrame()
    data = accepted_summary.copy()
    selected = data['primary_verified'].fillna(False).astype(bool)
    selected &= pd.to_numeric(data['accepted_v0_uM_s'], errors='coerce') > 0
    selected &= pd.to_numeric(data['pyruvate_mM'], errors='coerce') > 0
    return data.loc[selected].copy()


def _plot_accepted_velocities(fig, ax, accepted_summary: pd.DataFrame | None,
                              primary_fits: dict | None):
    """A real fitted accepted-v0 vs pyruvate plot, not the early-window plot."""
    primary_fits = primary_fits or {}
    data = _primary_rows(accepted_summary)
    if data.empty:
        ax.axis('off')
        ax.text(.5, .53, 'No time courses accepted for primary kinetic analysis.',
                ha='center', va='center', fontsize=13, transform=ax.transAxes)
        ax.text(.5, .44, 'Accept suitable velocities to generate this fit.',
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        return
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#555555']
    symbols = ['o', 's', 'D', '^', 'v']
    for idx, (enzyme, group) in enumerate(sorted(data.groupby('enzyme'), key=lambda x: str(x[0]))):
        c = palette[idx % len(palette)]
        group = group.sort_values('pyruvate_mM')
        ax.scatter(group.pyruvate_mM, group.accepted_v0_uM_s, s=42,
                   marker=symbols[idx % len(symbols)], color=c,
                   label=f'{enzyme}: accepted v0', zorder=3)
        ft = primary_fits.get(enzyme)
        if ft:
            xx = np.linspace(0, float(group.pyruvate_mM.max()), 450)
            ax.plot(xx, substrate_inhibition(xx, ft['vmax_uM_s'],
                    ft['km_mM'], ft['ki_mM']), color=c, lw=2.2,
                    label=f'{enzyme}: accepted-v0 fit')
    ax.set(xlabel='[Pyruvate] (mM)', ylabel='Accepted initial velocity (µM/s)')
    ax.grid(alpha=.16)
    ax.legend(loc='best', fontsize=9)


def _combined_parameter_rows(primary_fits: dict | None,
                             substrate_fits: dict | None,
                             selected_enzyme: str | None):
    """One comparative table: accepted fit plus both exploratory window fits."""
    primary_fits = primary_fits or {}
    substrate_fits = substrate_fits or {}
    rows = []
    enzymes = sorted(set(primary_fits) | ({str(selected_enzyme)} if selected_enzyme else set()))
    if not enzymes:
        enzymes = ['WT']
    for enzyme in enzymes:
        ft = primary_fits.get(enzyme)
        rows.append((enzyme, 'Accepted v0 (primary)', ft))
    for key, label in (('v_early_uM_s', 'Early window (exploratory)'),
                       ('v_late_uM_s', 'Late window (exploratory)')):
        rows.append((str(selected_enzyme or enzymes[0]), label, substrate_fits.get(key)))
    return rows


def _draw_combined_parameter_table(ax, primary_fits: dict | None,
                                   substrate_fits: dict | None,
                                   selected_enzyme: str | None,
                                   *, bbox=(.025, .64, .95, .25)):
    rows = _combined_parameter_rows(primary_fits, substrate_fits, selected_enzyme)
    cell_rows = []
    warnings = []
    def fmt(ft, key, n=4):
        val = ft.get(key, float('nan'))
        try:
            val = float(val)
            return f'{val:.{n}g}' if np.isfinite(val) else '—'
        except (TypeError, ValueError):
            return '—'
    for enzyme, label, ft in rows:
        if ft:
            cell_rows.append([str(enzyme)[:14], label, fmt(ft,'enzyme_concentration_nM',5),
                              fmt(ft,'vmax_uM_s',5), fmt(ft,'kcat_s_1',5),
                              fmt(ft,'km_mM',5), fmt(ft,'ki_mM',5),
                              fmt(ft,'r2',4), str(ft.get('n',''))])
            if ft.get('warning'):
                warnings.append(f'{enzyme} {label}: {ft["warning"]}')
        else:
            cell_rows.append([str(enzyme)[:14], label, '—', 'Not fitted', '—', '—', '—', '—', '—'])
    col_names = ['Enzyme','Analysis','[E] nM','Vmax µM/s','kcat s^-1',
                 'KM mM','KI mM','R²','n']
    x,y,w,h=bbox
    # More rows need additional table height; use available space at the top.
    height=max(h, min(.48, .084 + .045 * len(cell_rows)))
    tab = ax.table(cellText=cell_rows, colLabels=col_names,
                   colWidths=[.105,.225,.075,.11,.11,.10,.10,.085,.055],
                   loc='upper center', bbox=[x, y+h-height, w, height],
                   cellLoc='center', colLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.25 if len(cell_rows)<8 else 7.2)
    for (i,j), cell in tab.get_celld().items():
        cell.set_edgecolor('#D8DEE2')
        if i == 0:
            cell.set_facecolor('#E6EFF5'); cell.set_text_props(weight='bold')
        elif i == 1:
            cell.set_facecolor('#EAF5F1')
        elif i % 2 == 0:
            cell.set_facecolor('#F7FAFC')
    return warnings


def make_pdf(name: str, df: pd.DataFrame, fit_models: dict,
             early: dict, late: dict, table: pd.DataFrame,
             substrate_fits: dict, settings: dict,
             accepted_summary: pd.DataFrame | None = None,
             primary_fits: dict | None = None,
             selected_enzyme: str | None = None) -> bytes:
    output = BytesIO()
    t, a = df['time_s'].to_numpy(), df['absorbance'].to_numpy()
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(10.5, 7.6))
        fig.suptitle(f'LDHA biphasic time-course: {name}', fontsize=16, weight='bold')
        ax.scatter(t, a, s=10, alpha=.6, label='Measured A340', color='#444444')
        for label, data, col in [('Early linear', early, '#0072B2'),
                                 ('Late linear', late, '#D55E00')]:
            tt = np.linspace(data['start_s'], data['end_s'], 60)
            ax.plot(tt, data['intercept_A'] + data['slope_A_s']*tt,
                    lw=2.2, label=label, color=col)
            ax.axvspan(data['start_s'], data['end_s'], color=col, alpha=.10)
        tr = fit_models['transition']; tm = fit_models['t']
        ax.plot(tm, tr['predicted_A'], lw=1.8, ls='--', color='#009E73',
                label='Smooth exponential transition')
        ax.axvspan(0, float(settings.get('Mixing cutoff for selected trace (s)', 15)), color='#AAAAAA', alpha=.1)
        ax.set(xlabel='Time (s)', ylabel='Absorbance at 340 nm')
        ax.legend(loc='best'); ax.grid(alpha=.15)
        subtitle = (f"Early slope {early['slope_A_s']:.5g} A/s;  "
                    f"Late slope {late['slope_A_s']:.5g} A/s;  "
                    f"k_obs {tr['k_obs_s_1']:.3g} s⁻¹")
        fig.text(.08, .045, subtitle, fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        fig, axs = plt.subplots(2, 1, figsize=(10.5, 7.6),
                                gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle('Velocity versus pyruvate — descriptive fits', fontsize=15, weight='bold')
        for key, color, label in [('v_early_uM_s', '#0072B2', 'Early phase'),
                                  ('v_late_uM_s', '#D55E00', 'Late phase')]:
            sub = table[['pyruvate_mM', key]].dropna().sort_values('pyruvate_mM')
            sub = sub[sub['pyruvate_mM'] > 0]
            if len(sub):
                axs[0].scatter(sub.pyruvate_mM, sub[key], label=label, color=color)
            ft = substrate_fits.get(key)
            if ft and len(sub):
                x = np.linspace(sub.pyruvate_mM.min(), sub.pyruvate_mM.max(), 300)
                axs[0].plot(x, substrate_inhibition(x, ft['vmax_uM_s'],
                            ft['km_mM'], ft['ki_mM']), color=color, lw=2)
        axs[0].set(xlabel='[Pyruvate] (mM)', ylabel='Rate of NADH consumption (μM/s)')
        axs[0].legend(); axs[0].grid(alpha=.15)
        valid = table[['pyruvate_mM', 'v_late_over_early']].dropna().sort_values('pyruvate_mM')
        if len(valid):
            axs[1].scatter(valid.pyruvate_mM, valid.v_late_over_early, color='#CC79A7')
        axs[1].axhline(1, color='#444444', ls=':')
        axs[1].set(xlabel='[Pyruvate] (mM)', ylabel='Late / early velocity')
        axs[1].grid(alpha=.15)
        fig.tight_layout(rect=[0, .07, 1, .94])
        fig.text(.08, .026, 'Exploratory: two slopes do not establish the adduct mechanism.', fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # Primary accepted rates are a distinct analysis from the early/late exploratory curves.
        fig, ax = plt.subplots(figsize=(10.5, 7.6))
        fig.suptitle('Accepted initial-velocity substrate-inhibition fit (PRIMARY)',
                     fontsize=14, weight='bold')
        _plot_accepted_velocities(fig, ax, accepted_summary, primary_fits)
        fig.text(.08, .05, 'Uses only velocities explicitly accepted for primary kinetic analysis.', fontsize=9)
        fig.text(.08, .025, 'The fitted starting velocity after mixing is not necessarily the true t = 0 rate.', fontsize=9)
        fig.subplots_adjust(left=.12, right=.94, top=.88, bottom=.15)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.5, 7.6)); ax.axis('off')
        fig.suptitle('All kinetic parameters | ONE comparison table', fontsize=15, weight='bold')
        warning_list = _draw_combined_parameter_table(
            ax, primary_fits, substrate_fits, selected_enzyme,
            bbox=(.035, .73, .93, .165))
        ax.text(.06, .645, 'Green = PRIMARY: researcher-accepted v0; other rows use exploratory early/late windows.',
                transform=ax.transAxes, fontsize=9)
        ax.text(.06, .613, 'Model: v = Vmax[S] / (KM + [S] + [S]²/KI). All kinetic parameters are apparent.',
                transform=ax.transAxes, fontsize=9)
        ax.text(.06, .581, 'kcat = 1000 × Vmax (µM/s) / [E] active-site (nM). No parameter uncertainties estimated.',
                transform=ax.transAxes, fontsize=9)
        if warning_list:
            ax.text(.06, .55, 'Fit warnings: ' + ' | '.join(warning_list)[:145],
                    transform=ax.transAxes, fontsize=7.8, color='#A33D2B')
        description = [f'{k}: {v}' for k,v in settings.items()]
        description += ['', 'Interpretation notes:',
                        '• The early and late slopes depend on selected time windows.',
                        '• A smooth transition is not proof of adduct formation.',
                        '• The 10% rule applies to initial-rate interpretation, not exclusion of late phases.',
                        '• Pyruvate/background absorbance complicates absolute A340 and depletion.',
                        '• NADH depletion, NAD+ accumulation, and mixing can alter progress curves.',
                        '• Apparent Km and Ki may vary with the fitting window and model.',
                        '• Confirm findings using controls and independent replicate assays.']
        ax.text(.06,.500,'\n'.join(description), va='top', fontsize=9.2,
                family='DejaVu Sans', linespacing=1.45, transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)
    return output.getvalue()


def make_all_traces_pdf(parsed: dict, fits: dict, summary: pd.DataFrame,
                        settings: dict, show_exponential: bool = False,
                        show_segmented: bool = False) -> bytes:
    """Export all included A340 time courses as 2 x 2 landscape pages.

    The PDF reuses the per-file early/late windows calculated by the app and
    keeps one common x-axis range across traces, with independent y-axis scales.
    A review warning does not suppress the plot, but a failed fit is labeled.
    """
    if summary.empty:
        raise ValueError('No included time courses to export.')
    output = BytesIO()
    meta = summary.copy()
    meta['sort_conc'] = pd.to_numeric(meta['pyruvate_mM'], errors='coerce')
    meta['sort_conc'] = meta['sort_conc'].fillna(float('inf'))
    meta = meta.sort_values(['sort_conc', 'enzyme', 'file'], kind='stable')
    meta = meta[meta['file'].isin(parsed)].reset_index(drop=True)
    if meta.empty:
        raise ValueError('No uploaded time courses are available for the PDF.')

    global_end = max(float(parsed[str(row.file)]['df'].time_s.max())
                     for row in meta.itertuples(index=False))
    n_pages = int(np.ceil(len(meta) / 4))
    with PdfPages(output) as pdf:
        for page_number, start in enumerate(range(0, len(meta), 4), start=1):
            group = meta.iloc[start:start + 4]
            # The final partially filled page uses two wider plots rather than
            # leaving two empty graph slots.
            if len(group) <= 2:
                fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
            else:
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            axes = np.atleast_1d(axes).flatten()
            fig.suptitle('LDHA | All time courses', fontsize=16, weight='bold', y=.982)
            fig.text(.5, .941,
                     f'Ordered by pyruvate concentration  |  Page {page_number} of {n_pages}'
                     '  |  Independent vertical scales',
                     ha='center', fontsize=9, color='#444444')
            for ax, (_, row) in zip(axes, group.iterrows()):
                name = str(row['file'])
                df = parsed[name]['df']
                fit = fits.get(name, {})
                t = df['time_s'].to_numpy(float)
                y = df['absorbance'].to_numpy(float)
                cutoff = float(fit.get('mixing_cutoff_s', 15.0))
                ax.axvspan(0, cutoff, color='#AAAAAA', alpha=.12, linewidth=0)
                ax.plot(t, y, color='#555555', linewidth=1.0, label='Measured A340',
                        marker='.', markersize=2.4)
                for label, key, color in [('Early linear', 'early', '#0072B2'),
                                          ('Late linear', 'late', '#D55E00')]:
                    line_fit = fit.get(key)
                    if line_fit:
                        tx = np.linspace(line_fit['start_s'], line_fit['end_s'], 80)
                        ax.axvspan(line_fit['start_s'], line_fit['end_s'],
                                   color=color, alpha=.085, linewidth=0)
                        ax.plot(tx, line_fit['intercept_A'] +
                                line_fit['slope_A_s'] * tx,
                                color=color, linewidth=2.0, label=label)
                fitted_models = fit.get('models')
                if fitted_models:
                    for enabled, key, label, color, style in [
                        (show_exponential, 'transition', 'Exponential', '#009E73', '--'),
                        (show_segmented, 'segmented', 'Segmented', '#CC79A7', ':')]:
                        if enabled:
                            ax.plot(fitted_models['t'], fitted_models[key]['predicted_A'],
                                    color=color, lw=1.25, ls=style, label=label)
                concentration = pd.to_numeric(row['pyruvate_mM'], errors='coerce')
                s_text = f'{concentration:.4g} mM' if np.isfinite(concentration) else '[S] unknown'
                enzyme = str(row['enzyme'])
                issue = str(row['issue']) if pd.notna(row['issue']) else ''
                status = '  [REVIEW]' if issue.strip() else ''
                ax.set_title(f'{s_text}  |  {enzyme}{status}\n{name}', fontsize=9,
                             pad=5, color=('#9A3412' if issue.strip() else '#222222'))
                ax.set(xlim=(0, global_end), xlabel='Time (s)', ylabel='A340')
                ax.grid(alpha=.17)
                ax.tick_params(labelsize=8)
                try:
                    v1 = float(row['v_early_uM_s'])
                    v2 = float(row['v_late_uM_s'])
                    if np.isfinite(v1) and np.isfinite(v2):
                        fig_text = f'Early {v1:.4g}   |   Late {v2:.4g}  µM/s'
                    else:
                        fig_text = 'Could not calculate both velocities'
                except (TypeError, ValueError, KeyError):
                    fig_text = 'Could not calculate both velocities'
                fig_text += f'  |  mixing cutoff {cutoff:g} s'
                if str(row.get('acceptance_status', '')) == 'Accepted':
                    chosen = str(row.get('accepted_method', ''))
                    fig_text += f'  |  accepted: {chosen[:19]}'
                ax.text(.02, .03, fig_text, transform=ax.transAxes,
                        fontsize=8, ha='left', va='bottom',
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.82,'pad':1.5})

            for ax in axes[len(group):]:
                ax.axis('off')
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=min(5,len(labels)),
                       bbox_to_anchor=(.5, .919), frameon=False, fontsize=9)
            fig.text(.045, .038,
                     'Descriptive fits only. Initial-rate criteria do not exclude later phases; examine postmix absorbance and pyruvate blanks.',
                     fontsize=8, color='#555555')
            fig.text(.965, .038,
                     'Mixing cutoff adjustable for each trace',
                     ha='right', fontsize=8, color='#555555')
            fig.subplots_adjust(left=.075, right=.975, top=.80, bottom=.10,
                                wspace=.26, hspace=.42)
            pdf.savefig(fig, bbox_inches=None)
            plt.close(fig)
    return output.getvalue()


def make_acceptance_pdf(summary: pd.DataFrame, primary_fits: dict,
                        settings: dict | None = None,
                        substrate_fits: dict | None = None,
                        selected_enzyme: str | None = None) -> bytes:
    """Export explicit decisions for every trace and the selected primary v0 curves.

    Separate from the legacy 3-page exploratory report. PDF output preserves
    the chosen method, the researcher's kinetic/descriptive/exclude decision, and notes.
    """
    from textwrap import wrap

    if summary.empty:
        raise ValueError('No time courses were supplied.')
    output = BytesIO()
    settings = settings or {}
    columns = ['File', '[Pyr] mM', 'Screening', 'Decision', 'Method', 'v0 (uM/s)', 'In v0?']

    with PdfPages(output) as pdf:
        # Table pages; 12 per page keeps filenames and method names readable.
        rowcount = 12
        pages = int(np.ceil(len(summary) / rowcount))
        for page, start in enumerate(range(0, len(summary), rowcount), start=1):
            subset = summary.iloc[start:start + rowcount]
            fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
            fig.suptitle('LDHA | Accepted time-course analyses', fontsize=16, fontweight='bold', y=.96)
            fig.text(.05, .90, f'Analysis decisions | page {page} of {pages} | '
                     'Unaccepted or stale decisions are NOT primary velocities', fontsize=9)
            records = []
            for _, r in subset.iterrows():
                def trunc(x, n):
                    text = str(x) if pd.notna(x) else ''
                    return text if len(text) <= n else text[:n-1] + '…'
                conc = pd.to_numeric(r.get('pyruvate_mM'), errors='coerce')
                v = pd.to_numeric(r.get('accepted_v0_uM_s'), errors='coerce')
                outcome_short = {
                    'No detectable time-dependent change': 'Linear-like',
                    'Detectable time-dependent change': 'Time change',
                    'Transition rate unresolved': 'Rate unresolved',
                    'Not assessed': 'Not assessed',
                }.get(str(r.get('timecourse_outcome', '')), 'Unclassified')
                records.append([trunc(r.get('file',''), 29),
                                f'{conc:.4g}' if pd.notna(conc) else '',
                                outcome_short,
                                {'Accept for kinetic analysis': 'Kinetic',
                                 'Accept for descriptive analysis only': 'Descriptive',
                                 'Exclude trace': 'Excluded'}.get(
                                    str(r.get('review_decision', '')), 'Not reviewed'),
                                trunc(r.get('accepted_method', ''), 27),
                                f'{v:.4g}' if pd.notna(v) else '',
                                'YES' if r.get('primary_verified', False) else 'no'])
            tab = ax.table(cellText=records, colLabels=columns, cellLoc='left', colLoc='left',
                           colWidths=[.26,.08,.13,.15,.21,.09,.08],
                           loc='upper left', bbox=[.02, .15, .96, .70])
            tab.auto_set_font_size(False); tab.set_fontsize(8)
            tab.scale(1,1.18)
            for (i,j), cell in tab.get_celld().items():
                cell.set_edgecolor('#D8DEE2')
                if i == 0:
                    cell.set_facecolor('#EBF1F5'); cell.set_text_props(weight='bold')
                elif i % 2 == 0:
                    cell.set_facecolor('#F8FAFC')
            fig.text(.055,.11,'Primary = researcher selected kinetic analysis. Review initial-rate suitability; this is not automated proof.',fontsize=9)
            fig.text(.055,.08,'Screening: Linear-like = no detectable rate change; Time change = supported curvature;',fontsize=8.5)
            fig.text(.055,.055,'Rate unresolved = curvature but poorly constrained k. Exponential initial = first postmix sample.',fontsize=8.5)
            pdf.savefig(fig); plt.close(fig)

        # Main analysis plot for each enzyme represented by researcher-selected velocities.
        valid = summary.copy()
        mask = (valid['primary_verified'].fillna(False).astype(bool)
                & (pd.to_numeric(valid['accepted_v0_uM_s'], errors='coerce') > 0)
                & (pd.to_numeric(valid['pyruvate_mM'], errors='coerce') > 0))
        valid = valid.loc[mask]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.suptitle('LDHA | Primary accepted initial velocities', fontsize=16, weight='bold', y=.96)
        if valid.empty:
            ax.axis('off')
            ax.text(.5,.54,'No velocities selected for primary kinetic analysis yet.',
                    ha='center', va='center', fontsize=13, transform=ax.transAxes)
        else:
            for idx, (enzyme, group) in enumerate(valid.groupby('enzyme')):
                mark = ['o','s','D','^','v','P'][idx % 6]
                points = group.sort_values('pyruvate_mM')
                ax.scatter(points.pyruvate_mM, points.accepted_v0_uM_s,
                           s=44, marker=mark, label=f'{enzyme}: accepted v0')
                ft = primary_fits.get(enzyme)
                if ft:
                    x = np.linspace(0, float(points.pyruvate_mM.max()), 350)
                    ax.plot(x, substrate_inhibition(x, ft['vmax_uM_s'],
                            ft['km_mM'], ft['ki_mM']), lw=2,
                            label=f'{enzyme}: descriptive inhibition fit')
            ax.set(xlabel='[Pyruvate] (mM)', ylabel='Accepted v0 (uM/s)')
            ax.grid(alpha=.16); ax.legend(fontsize=9)
        fig.text(.07,.085,'Curve fit is descriptive; mixed methods and postmix extrapolation require scientific review.',fontsize=9)
        fig.subplots_adjust(left=.12, right=.95, top=.88, bottom=.17)
        pdf.savefig(fig); plt.close(fig)

        # Unified table: accepted primary and exploratory early/late fits.
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
        fig.suptitle('LDHA | All fitted kinetic parameters (comparison)',
                     fontsize=16, fontweight='bold', y=.96)
        fit_warnings = _draw_combined_parameter_table(
            ax, primary_fits, substrate_fits, selected_enzyme,
            bbox=(.035, .65, .93, .20))
        ax.text(.055, .60, 'PRIMARY accepted-v0 fit is highlighted. Early/late windows are exploratory comparisons.',
                fontsize=9.5, transform=ax.transAxes)
        ax.text(.055, .56, 'For multiple enzymes, early/late exploratory rows refer to the selected enzyme.',
                fontsize=9, transform=ax.transAxes)
        ax.text(.055, .48, 'Model: v = Vmax[S] / (KM + [S] + [S]²/KI)',
                fontsize=10, transform=ax.transAxes)
        ax.text(.055, .44, 'kcat = 1000 × Vmax (µM/s) / [E] active-site (nM). Values are apparent.',
                fontsize=9, transform=ax.transAxes)
        ax.text(.055, .40, 'At least five distinct positive substrate concentrations are required for a fit.',
                fontsize=9, transform=ax.transAxes)
        ax.text(.055, .36, 'An accepted post-mixing rate may already reflect inhibition during the mixing period.',
                fontsize=9, transform=ax.transAxes)
        ax.text(.055, .32, 'Fit uncertainties are not estimated by the current fitting routine.',
                fontsize=9, transform=ax.transAxes)
        if fit_warnings:
            from textwrap import wrap as _wrap
            for i, line in enumerate(_wrap('Fit warnings: ' + '; '.join(fit_warnings), 110)[:3]):
                ax.text(.055, .24-i*.035, line, fontsize=8, color='#A33D2B', transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)

        # Preserve full reviewer notes and cautions, not just truncated table cells.
        paragraphs = ['Interpretation and selection record',
                      'The decision is specific to the current analysis settings and must be reviewed after changes.',
                      'Not reviewed, descriptive-only, and excluded traces cannot enter the primary dataset.',
                      'Actual NADH conversion is not established by A340 changes without matched blanks.',
                      'Exponential-model starting velocity is at first retained POSTMIX time, not t=0.',
                      '']
        paragraphs += [f'{key}: {value}' for key,value in settings.items()]
        paragraphs.append('')
        for _,r in summary.iterrows():
            paragraphs.append(f"{r['file']} | {r.get('timecourse_outcome', '')} | "
                              f"{r.get('review_decision', '') or r.get('acceptance_status', '')} | {r.get('accepted_method', '')}")
            if str(r.get('accepted_basis', '')).strip():
                paragraphs.append('Basis: ' + str(r['accepted_basis']))
            if str(r.get('accepted_caution', '')).strip():
                paragraphs.append('Caution: ' + str(r['accepted_caution']))
            if str(r.get('reviewer_note', '')).strip():
                paragraphs.append('Reviewer note: ' + str(r['reviewer_note']))
            paragraphs.append('')
        lines = []
        for para in paragraphs:
            lines.extend(wrap(para, width=112, break_long_words=False) or [''])
        for start in range(0, len(lines), 34):
            fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
            fig.suptitle('LDHA | Analysis notes and cautions', fontsize=15, weight='bold', y=.96)
            ax.text(.05,.89,'\n'.join(lines[start:start+34]), transform=ax.transAxes,
                    va='top', fontsize=9, family='DejaVu Sans', linespacing=1.55)
            pdf.savefig(fig); plt.close(fig)
    return output.getvalue()
