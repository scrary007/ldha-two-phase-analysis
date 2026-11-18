"""Scientific calculations for the LDHA biphasic time-course Streamlit app.

This module deliberately reports empirical fits; none establishes the molecular
identity of an inhibitory species.  Concentrations are in mM unless specified.
"""
from __future__ import annotations

import csv
from io import StringIO
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import linregress


class DataError(ValueError):
    """Raised when a raw instrument export cannot be interpreted."""


def _number(s: Any) -> float:
    v = str(s).strip().replace('\ufeff', '').replace('\u2212', '-')
    if not v:
        raise ValueError('empty')
    if ',' in v and '.' not in v:
        v = v.replace(',', '.')   # e.g., decimal commas in semicolon-separated files
    return float(v)


def read_cary_csv(blob: bytes, time_column: int = 1, absorbance_column: int = 2,
                  time_unit: str = 'seconds') -> pd.DataFrame:
    """Find the longest monotonic numeric table in a Cary-like CSV export.

    Columns are 1-indexed; manual indices make multiple Cary export layouts
    usable without assuming the first line is a header.  Cannot decode XLSX.
    """
    if time_column == absorbance_column or min(time_column, absorbance_column) < 1:
        raise DataError('Time and absorbance columns must differ and be positive.')
    text = None
    for encoding in ('utf-8-sig', 'utf-16', 'cp1252'):
        try:
            candidate = blob.decode(encoding)
            # UTF-16 auto-detection can accept garbage from arbitrary UTF-8 data;
            # skip if it produced embedded NULs.
            if '\x00' not in candidate:
                text = candidate
                break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DataError('Could not decode the file as text.')
    max_col = max(time_column, absorbance_column)
    possibilities = []
    for delimiter in (',', '\t', ';'):
        rows = csv.reader(StringIO(text), delimiter=delimiter)
        run: list[tuple[float, float]] = []
        runs = []
        for row in rows:
            try:
                if len(row) < max_col:
                    raise ValueError('missing selected columns')
                t = _number(row[time_column - 1]); y = _number(row[absorbance_column - 1])
                if not np.isfinite(t) or not np.isfinite(y):
                    raise ValueError('nonfinite')
                if run and t < run[-1][0]:
                    # Cary may output multiple spectra; only fit a time-ordered block.
                    if len(run) >= 6:
                        runs.append(run)
                    run = []
                run.append((t, y))
            except (ValueError, TypeError, IndexError):
                if len(run) >= 6:
                    runs.append(run)
                run = []
        if len(run) >= 6:
            runs.append(run)
        if runs:
            best = max(runs, key=len)
            possibilities.append(best)
    if not possibilities:
        raise DataError('Could not find at least 6 consecutive numeric time/absorbance rows. '
                        'Try the time/absorbance column selectors or export as CSV.')
    rows = max(possibilities, key=len)
    data = pd.DataFrame(rows, columns=['time_s', 'absorbance'])
    if time_unit == 'minutes':
        data['time_s'] *= 60.0
    elif time_unit != 'seconds':
        raise DataError('Time unit must be seconds or minutes.')
    data = data.groupby('time_s', as_index=False, sort=True)['absorbance'].mean()
    if len(data) < 6 or data['time_s'].nunique() < 6:
        raise DataError('Not enough distinct time points.')
    return data


def filename_pyruvate(name: str) -> float | None:
    """Recognize 53.33mM and 53,3mM patterns, not dates in filenames."""
    values = re.findall(r'(?<![\d.])([0-9]+(?:[.,][0-9]+)?)\s*mM(?=$|[^A-Za-z0-9])', name,
                        flags=re.IGNORECASE)
    return float(values[-1].replace(',', '.')) if values else None


def signal_slope_to_velocity(m_abs_per_s: float, extinction_mM: float = 6.22,
                             path_cm: float = 1.0, blank_slope: float = 0.0) -> float:
    """NADH consumption, μM/s.  Positive values represent declining A340."""
    if extinction_mM <= 0 or path_cm <= 0:
        raise ValueError('Extinction coefficient and pathlength must be positive.')
    return -1000.0 * (m_abs_per_s - blank_slope) / (extinction_mM * path_cm)


def _metrics(y: np.ndarray, pred: np.ndarray, params: int) -> dict:
    n = len(y)
    rss = float(np.sum((y - pred) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    aic = n * np.log(max(rss / n, 1e-300)) + 2 * params
    aicc = aic + 2 * params * (params + 1) / (n - params - 1) if n > params + 1 else np.nan
    return {'n': n, 'r2': 1 - rss / tss if tss > 0 else np.nan,
            'rmse_A': float(np.sqrt(rss / n)), 'aicc': float(aicc), 'rss': rss}


def fit_line(time: np.ndarray, a: np.ndarray, start_s: float, end_s: float,
             min_points: int = 5) -> dict:
    mask = (time >= start_s) & (time <= end_s)
    t = time[mask]; y = a[mask]
    if len(t) < min_points or np.ptp(t) <= 0:
        raise DataError(f'Need at least {min_points} distinct points within '
                        f'{start_s:g}–{end_s:g} seconds; found {len(t)}.')
    lr = linregress(t, y)
    pred = lr.intercept + lr.slope * t
    return {'start_s': float(t[0]), 'end_s': float(t[-1]), 'n': len(t),
            'slope_A_s': float(lr.slope), 'intercept_A': float(lr.intercept),
            'slope_se_A_s': float(lr.stderr), 'r2': float(lr.rvalue**2),
            'rmse_A': float(np.sqrt(np.mean((y - pred)**2)))}


def transition_signal(t: np.ndarray, a0: float, m_initial: float,
                      m_final: float, k_obs: float) -> np.ndarray:
    """A(t)=A0 + m_f*t + (m_i-m_f)*(1-exp(-kt))/k, t >= 0."""
    t = np.asarray(t, dtype=float)
    if k_obs <= 0:
        return a0 + m_initial * t
    return a0 + m_final * t + (m_initial - m_final) * (-np.expm1(-k_obs*t)) / k_obs


def fit_transition(t: np.ndarray, a: np.ndarray) -> dict:
    """Smooth empirical slope transition. k_obs is NOT an adduct rate constant."""
    if len(t) < 12:
        raise DataError('At least 12 points needed for the continuous transition fit.')
    x = t - t[0]
    duration = float(x[-1])
    if duration <= 0:
        raise DataError('Time span is zero.')
    n_edge = max(5, len(t) // 7)
    mi = float(np.polyfit(x[:n_edge], a[:n_edge], 1)[0])
    mf = float(np.polyfit(x[-n_edge:], a[-n_edge:], 1)[0])
    median_dt = float(np.median(np.diff(x)))
    k_min = 1 / (100 * duration)
    k_max = min(100, 10 / max(median_dt, 1e-6))
    k_starts = [0.2 / duration, 1 / duration, 4 / duration, 12 / duration]
    best = None
    for start in k_starts:
        guess = [float(a[0]), mi, mf, np.clip(start, k_min*1.01, k_max*.99)]
        result = least_squares(lambda p: transition_signal(x, *p) - a,
                               x0=guess,
                               bounds=([-np.inf, -np.inf, -np.inf, k_min],
                                       [np.inf, np.inf, np.inf, k_max]),
                               max_nfev=5000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    p = best.x
    pred = transition_signal(x, *p)
    return {'a0': float(p[0]), 'm_initial_A_s': float(p[1]),
            'm_final_A_s': float(p[2]), 'k_obs_s_1': float(p[3]),
            't_half_s': float(np.log(2) / p[3]),
            'k_at_bound': bool(p[3] <= k_min * 1.02 or p[3] >= k_max * .98),
            'k_bound_side': ('lower' if p[3] <= k_min * 1.02
                             else 'upper' if p[3] >= k_max * .98 else 'none'),
            'success': bool(best.success), 'predicted_A': pred,
            **_metrics(a, pred, params=4)}


def classify_timecourse(models: dict, min_aicc_improvement: float = 10.0,
                        min_relative_slope_change: float = 0.10) -> dict:
    """Exploratory screening of post-mixing progress-curve shapes.

    The thresholds are heuristic (not a hypothesis test): a nonlinear model
    must improve AICc by >=10 *and* show >=10% relative slope change before
    the program calls the curvature meaningful. High-frequency assay points
    may be serially correlated, so this must not be treated as evidence for
    any specific inhibition mechanism or as a quantitative confidence level.

    A fitted exponential k is interpreted only when exponential curvature is
    supported and k is not at its fitting bound. A segmented fit alone can
    suggest changing rates without establishing an exponential transition.
    """
    linear, transition, segmented = (models[k] for k in
                                     ('linear', 'transition', 'segmented'))
    def relative_change(fit):
        a, b = float(fit['m_initial_A_s']), float(fit['m_final_A_s'])
        return abs(b - a) / max(abs(a), abs(b), 1e-12)
    exp_gain = float(linear['aicc'] - transition['aicc'])
    seg_gain = float(linear['aicc'] - segmented['aicc'])
    exp_fraction = relative_change(transition)
    seg_fraction = relative_change(segmented)
    exp_supported = (np.isfinite(exp_gain) and exp_gain >= min_aicc_improvement
                     and exp_fraction >= min_relative_slope_change)
    seg_supported = (np.isfinite(seg_gain) and seg_gain >= min_aicc_improvement
                     and seg_fraction >= min_relative_slope_change)
    side = transition.get('k_bound_side', 'bound' if transition['k_at_bound'] else 'none')
    if not exp_supported and not seg_supported:
        label = 'No detectable time-dependent change'
        reason = ('Neither nonlinear model meets both exploratory screening criteria '
                  '(AICc improvement ≥10 and modeled slope change ≥10%). '
                  'An exponential k at its bound is not meaningful when the trace is effectively linear.')
        next_step = ('Use a single post-mixing linear window for an initial velocity if the '
                     'initial-rate assumptions are satisfied; keep this trace as a low-pyruvate '
                     'comparison. Do not interpret k_obs. Check residuals and replicates.')
    elif exp_supported and not transition['k_at_bound']:
        label = 'Detectable time-dependent change'
        reason = ('The exponential model meets the exploratory evidence and effect-size '
                  'criteria, and k_obs did not reach a fitting bound. This is a change '
                  'in rate, not proof of inhibition or adduct formation.')
        next_step = ('Inspect residuals, verify the cutoff and early/late windows, and '
                     'repeat the experiment. Test for NADH depletion, product accumulation '
                     'and pyruvate-dependent absorbance before comparing k_obs across [pyruvate].')
    else:
        label = 'Transition rate unresolved'
        if exp_supported and transition['k_at_bound']:
            why = f'The exponential k_obs is at its {side} fitting bound.'
        else:
            why = 'The segmented fit suggests a changing slope, but the exponential rate is not well supported.'
        reason = why + ' Curvature may be present, but k_obs should not be interpreted.'
        next_step = ('Inspect the trace and mixing cutoff. For a possible rapid event, '
                     'reduce mixing dead time or use faster acquisition; for a slow event, '
                     'collect a longer time course while monitoring depletion. Repeat and '
                     'compare with matched controls. Do not report a quantitative k_obs.')
    return {'label': label, 'reason': reason, 'next_step': next_step,
            'exponential_aicc_gain': exp_gain, 'segmented_aicc_gain': seg_gain,
            'exponential_relative_slope_change': exp_fraction,
            'segmented_relative_slope_change': seg_fraction,
            'k_bound_side': side,
            'k_interpretable': label == 'Detectable time-dependent change'}


def fit_segmented(t: np.ndarray, a: np.ndarray, min_points_each: int = 6) -> dict:
    """Continuous 2-line model with breakpoint found by RSS grid search."""
    if len(t) < 2 * min_points_each + 1:
        raise DataError('Not enough points to locate a breakpoint.')
    x = t - t[0]
    best = None
    # Constrain to real observed times, preserving at least min_points_each per side.
    for i in range(min_points_each - 1, len(x) - min_points_each):
        breakpoint = x[i]
        h = np.maximum(x - breakpoint, 0.0)
        X = np.column_stack([np.ones_like(x), x, h])
        coef, *_ = np.linalg.lstsq(X, a, rcond=None)
        pred = X @ coef
        rss = np.sum((a - pred)**2)
        if best is None or rss < best[0]:
            best = (rss, breakpoint, coef, pred)
    _, bp, coef, pred = best
    return {'break_s': float(bp + t[0]), 'm_initial_A_s': float(coef[1]),
            'm_final_A_s': float(coef[1] + coef[2]),
            'predicted_A': pred, **_metrics(a, pred, params=4)}


def apparent_postmix_fraction(t: np.ndarray, absorbance: np.ndarray,
                              start_s: float, end_s: float,
                              extinction_mM: float, path_cm: float,
                              initial_nadh_uM: float) -> float:
    """Absorbance drop as an *apparent* NADH-equivalent fraction of input NADH.

    This is NOT a quantitative NADH depletion estimate when pyruvate and
    products contribute a changing A340 signal. Only use after mixing. The
    amount consumed before the mixing cutoff is unknown.
    """
    t = np.asarray(t, float)
    a = np.asarray(absorbance, float)
    if not (len(t) >= 2 and t[0] <= start_s <= end_s <= t[-1]):
        return float('nan')
    if min(extinction_mM, path_cm, initial_nadh_uM) <= 0:
        return float('nan')
    drop_A = float(np.interp(start_s, t, a) - np.interp(end_s, t, a))
    return max(0.0, drop_A * 1000 / (extinction_mM * path_cm * initial_nadh_uM))


def fit_all_models(df: pd.DataFrame, dead_time_s: float = 15.0) -> dict:
    df = df[df.time_s >= dead_time_s]
    if len(df) < 13:
        raise DataError('Too few points after excluded mixing/dead-time period.')
    t = df.time_s.to_numpy(float); a = df.absorbance.to_numpy(float)
    reg = linregress(t, a)
    straight_pred = reg.intercept + reg.slope * t
    linear = {'slope_A_s': float(reg.slope), 'intercept_A': float(reg.intercept),
              'predicted_A': straight_pred, **_metrics(a, straight_pred, 2)}
    return {'linear': linear, 'segmented': fit_segmented(t, a),
            'transition': fit_transition(t, a), 't': t, 'A': a}


def substrate_inhibition(s: np.ndarray, vmax: float, km: float,
                         ki: float) -> np.ndarray:
    """v = Vmax*S/(Km + S + S**2/Ki), all concentrations in mM."""
    s = np.asarray(s, float)
    return vmax * s / (km + s + s*s/ki)


def fit_substrate_inhibition(s: np.ndarray, v: np.ndarray) -> dict:
    """Independent, descriptive phase fit; not an equilibrium interpretation."""
    s = np.asarray(s, float); v = np.asarray(v, float)
    valid = np.isfinite(s) & np.isfinite(v) & (s > 0) & (v > 0)
    s, v = s[valid], v[valid]
    if len(s) < 5 or len(np.unique(s)) < 5:
        raise DataError('Need at least five positive, distinct [pyruvate] values and velocities.')
    span = float(np.max(s) / np.min(s))
    vmax0 = float(max(np.max(v), .001))
    mids = [float(np.median(s)), float(np.min(s)*3), float(np.max(s)/10)]
    kis = [float(np.max(s)), float(5*np.max(s)), float(np.max(s)/3)]
    best = None
    for km0 in mids:
        for ki0 in kis:
            p0 = np.log([vmax0, max(km0, 1e-9), max(ki0, 1e-9)])
            res = least_squares(lambda logp: substrate_inhibition(s, *np.exp(logp)) - v,
                                p0, bounds=(np.log([1e-10, 1e-9, 1e-6]),
                                            np.log([1e6, 1e6, 1e9])), max_nfev=4000)
            if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
                best = res
    vmax, km, ki = np.exp(best.x)
    pred = substrate_inhibition(s, vmax, km, ki)
    return {'vmax_uM_s': float(vmax), 'km_mM': float(km), 'ki_mM': float(ki),
            'warning': ('Estimated Ki exceeds 100× maximum tested [S]; high-[S] '
                        'inhibition is not resolved.' if ki > 100 * max(s) else ''),
            'n': len(s), 's_span': span, 'success': bool(best.success),
            **_metrics(v, pred, params=3)}


def recommended_windows(t: np.ndarray, cutoff_s: float = 15.0) -> dict[str, float]:
    """Propose early/late fitting windows AFTER a per-trace mixing cutoff.

    They are starting points only. The user should inspect and edit each trace.
    """
    t = np.asarray(t, float)
    lo, hi = float(np.min(t)), float(np.max(t))
    dt = float(np.median(np.diff(np.unique(t))))
    after = max(lo, float(cutoff_s))
    span = hi - after
    if span < max(10*dt, 1.0):
        raise DataError('Insufficient time points after the mixing cutoff to propose two windows.')
    early_start = after
    early_end = after + min(15.0, max(0.38*span, 5*dt))
    late_start = after + min(25.0, max(0.55*span, early_end-after+2*dt))
    late_end = after + min(55.0, max(0.95*span, late_start-after+5*dt))
    early_end = min(early_end,hi)
    late_start = min(late_start,hi)
    late_end = min(late_end,hi)
    return dict(early_start=float(early_start), early_end=float(early_end),
                late_start=float(late_start), late_end=float(late_end))
