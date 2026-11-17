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
                        '• NADH depletion, NAD+ accumulation, and mixing can alter progress curves.',
                        '• Apparent Km and Ki may vary with the fitting window and model.',
                        '• Confirm findings using controls and independent replicate assays.']
        ax.text(.08,.92,'\n'.join(description), va='top', fontsize=11,
                family='DejaVu Sans', linespacing=1.65, transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)
    return output.getvalue()
