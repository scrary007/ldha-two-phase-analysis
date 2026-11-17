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
        fig.suptitle('Analysis settings and limitations', fontsize=15, weight='bold')
        description = [f'{k}: {v}' for k,v in settings.items()]
        description += ['', 'Interpretation notes:',
                        '• The early and late slopes depend on selected time windows.',
                        '• A smooth transition is not proof of adduct formation.',
                        '• The 10% rule applies to initial-rate interpretation, not exclusion of late phases.',
                        '• Pyruvate/background absorbance complicates absolute A340 and depletion.',
                        '• NADH depletion, NAD+ accumulation, and mixing can alter progress curves.',
                        '• Apparent Km and Ki may vary with the fitting window and model.',
                        '• Confirm findings using controls and independent replicate assays.']
        ax.text(.08,.92,'\n'.join(description), va='top', fontsize=11,
                family='DejaVu Sans', linespacing=1.65, transform=ax.transAxes)
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
