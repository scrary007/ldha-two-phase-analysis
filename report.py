"""Portable PDF report for the LDHA biphasic app, including no raw data storage."""
from __future__ import annotations
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis import substrate_inhibition, transition_signal


def make_pdf(name: str, df: pd.DataFrame, fit_models: dict,
             early: dict, late: dict, table: pd.DataFrame,
             substrate_fits: dict, settings: dict) -> bytes:
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

        fig, ax = plt.subplots(figsize=(10.5, 7.6)); ax.axis('off')
        fig.suptitle('Analysis settings and fitted kinetic parameters', fontsize=15, weight='bold')
        # These fits describe the EARLY and LATE linear windows, respectively;
        # they do not combine with the separate accepted-initial-velocity dataset.
        phase_names = [('v_early_uM_s', 'Early window'),
                       ('v_late_uM_s', 'Late window')]
        param_rows = []
        for phase_key, phase_label in phase_names:
            ft = substrate_fits.get(phase_key)
            if ft:
                param_rows.append([phase_label,
                                   f"{ft['vmax_uM_s']:.4g}",
                                   f"{ft['km_mM']:.4g}",
                                   f"{ft['ki_mM']:.4g}",
                                   f"{ft.get('r2', float('nan')):.4g}",
                                   str(ft.get('n', ''))])
            else:
                param_rows.append([phase_label, 'Not fit', '—', '—', '—', '—'])
        header = ['Phase', 'Vmax (µM/s)', 'KM (mM)', 'KI (mM)', 'R²', 'n']
        tab = ax.table(cellText=param_rows, colLabels=header, cellLoc='center',
                       colLoc='center', loc='upper center', bbox=[.04, .72, .92, .13],
                       colWidths=[.22,.18,.17,.17,.13,.13])
        tab.auto_set_font_size(False); tab.set_fontsize(10)
        for (i,j), cell in tab.get_celld().items():
            cell.set_edgecolor('#D8DEE2')
            if i == 0: cell.set_facecolor('#EBF1F5'); cell.set_text_props(weight='bold')
        ax.text(.06, .68, 'Descriptive apparent parameters: v = Vmax[S] / (KM + [S] + [S]²/KI).',
                transform=ax.transAxes, fontsize=9)
        ax.text(.06, .645, 'kcat is not computed: enzyme active-site concentration is not entered. Fit errors are not estimated.',
                transform=ax.transAxes, fontsize=8.6)
        warnings = [f"{label}: {substrate_fits[key].get('warning')}"
                    for key,label in phase_names
                    if key in substrate_fits and substrate_fits[key].get('warning')]
        if warnings:
            ax.text(.06, .608, 'Warnings: ' + ' | '.join(warnings)[:145],
                    transform=ax.transAxes, fontsize=8, color='#A33D2B')
        description = [f'{k}: {v}' for k,v in settings.items()]
        description += ['', 'Interpretation notes:',
                        '• The early and late slopes depend on selected time windows.',
                        '• A smooth transition is not proof of adduct formation.',
                        '• The 10% rule applies to initial-rate interpretation, not exclusion of late phases.',
                        '• Pyruvate/background absorbance complicates absolute A340 and depletion.',
                        '• NADH depletion, NAD+ accumulation, and mixing can alter progress curves.',
                        '• Apparent Km and Ki may vary with the fitting window and model.',
                        '• Confirm findings using controls and independent replicate assays.']
        ax.text(.06,.555,'\n'.join(description), va='top', fontsize=9.2,
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
                        settings: dict | None = None) -> bytes:
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

        # Report parameter estimates independently for each enzyme/variant.
        # No fit can be calculated with fewer than five distinct substrate concentrations.
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis('off')
        fig.suptitle('LDHA | Accepted v0 kinetic parameters', fontsize=16,
                     fontweight='bold', y=.96)
        rows = []
        for enzyme_name, ft in sorted(primary_fits.items()):
            rows.append([str(enzyme_name)[:28],
                         f"{ft['vmax_uM_s']:.5g}",
                         f"{ft['km_mM']:.5g}",
                         f"{ft['ki_mM']:.5g}",
                         f"{ft.get('r2', float('nan')):.4g}",
                         str(ft.get('n', ''))])
        if rows:
            hdr = ['Enzyme', 'Vmax (µM/s)', 'KM (mM)', 'KI (mM)', 'R²', 'n']
            tab = ax.table(cellText=rows, colLabels=hdr, cellLoc='center',
                           colLoc='center', colWidths=[.32,.17,.15,.15,.12,.09],
                           loc='upper center', bbox=[.05, max(.30, .82-.055*len(rows)), .90,
                                                     min(.45, .075+.055*len(rows))])
            tab.auto_set_font_size(False); tab.set_fontsize(10)
            for (i,j), cell in tab.get_celld().items():
                cell.set_edgecolor('#D8DEE2')
                if i == 0: cell.set_facecolor('#EBF1F5'); cell.set_text_props(weight='bold')
            warnings = [f"{enzyme_name}: {ft.get('warning')}" for enzyme_name,ft in sorted(primary_fits.items())
                        if ft.get('warning')]
            if warnings:
                ax.text(.06, .25, 'Fit warnings: ' + '; '.join(warnings)[:180],
                        transform=ax.transAxes, fontsize=9, color='#A33D2B', wrap=True)
        else:
            ax.text(.06, .74, 'No primary substrate-inhibition fit available. At least five distinct',
                    transform=ax.transAxes, fontsize=11)
            ax.text(.06, .70, 'positive concentrations with accepted initial velocities are required.',
                    transform=ax.transAxes, fontsize=11)
        ax.text(.06,.15,'Model: v = Vmax[S] / (KM + [S] + [S]²/KI)', fontsize=10,
                transform=ax.transAxes)
        ax.text(.06,.11,'Apparent parameters only; enzyme concentration is not entered, so kcat is not calculated.',
                fontsize=9, transform=ax.transAxes)
        ax.text(.06,.075,'Fit uncertainties are not estimated by the current fitting routine.',
                fontsize=9, transform=ax.transAxes)
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
