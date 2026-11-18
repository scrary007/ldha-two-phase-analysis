"""Selection and acceptance of LDHA time-course analysis methods.

All automated interpretations are exploratory. The researcher's one-choice
decision controls which fitted velocities enter the primary v0 set.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math

import numpy as np

from analysis import DataError, apparent_postmix_fraction, fit_line, signal_slope_to_velocity

SINGLE = 'Single linear window'
EXPONENTIAL = 'Exponential transition'
EARLY = 'Early window (provisional)'
EXCLUDE = 'Exclude trace'
METHODS = (SINGLE, EXPONENTIAL, EARLY, EXCLUDE)

NOT_REVIEWED = 'Not reviewed'
KINETIC = 'Accept for kinetic analysis'
DESCRIPTIVE = 'Accept for descriptive analysis only'
EXCLUDED = 'Exclude trace'
DECISIONS = (NOT_REVIEWED, KINETIC, DESCRIPTIVE, EXCLUDED)


def available_decisions(candidate: dict | None) -> tuple[str, ...]:
    """A trace with an unresolved or invalid fit cannot silently enter the v0 plot."""
    if candidate is None or candidate.get('method') == EXCLUDE:
        return (NOT_REVIEWED, EXCLUDED)
    v = candidate.get('v0_uM_s')
    if v is None or not math.isfinite(v) or v <= 0:
        return (NOT_REVIEWED, EXCLUDED)
    if not candidate.get('eligible_for_primary', False):
        return (NOT_REVIEWED, DESCRIPTIVE, EXCLUDED)
    return DECISIONS


def recommended_decision(outcome: str, candidate: dict | None, issue: str = '') -> str:
    """Conservative BULK suggestion, not a confirmation of the inhibition mechanism.

    Only nearly linear, unflagged windows with <=10% postmix apparent NADH-
    equivalent change are suggested for kinetic analysis. Curved and unresolved
    traces remain descriptive, even if an analyst could manually select a
    provisional early velocity. The researcher must review the bulk choices.
    """
    options = available_decisions(candidate)
    if options == (NOT_REVIEWED, EXCLUDED):
        return EXCLUDED
    if (outcome == 'No detectable time-dependent change'
            and candidate.get('method') == SINGLE
            and KINETIC in options and not str(issue).strip()):
        fraction = candidate.get('apparent_fraction_initial_window')
        if fraction is not None and math.isfinite(fraction) and fraction <= .10:
            return KINETIC
    return DESCRIPTIVE


def suggest_method(outcome: str) -> str:
    return {
        'No detectable time-dependent change': SINGLE,
        'Detectable time-dependent change': EXPONENTIAL,
        'Transition rate unresolved': EARLY,
    }.get(outcome, EXCLUDE)


def analysis_fingerprint(inputs: dict) -> str:
    """Marks previously accepted analyses as stale when an input changes."""
    serialized = json.dumps(inputs, sort_keys=True, allow_nan=False, separators=(',', ':'), default=str)
    return sha256(serialized.encode('utf-8')).hexdigest()


def compute_candidate(method: str, t, a, fits: dict, *,
                      single_start_s: float, single_end_s: float,
                      cutoff_s: float, extinction: float, path_cm: float,
                      blank_slope: float, initial_nadh: float,
                      outcome: str) -> dict:
    """Compute a candidate velocity with explicit provenance and caveats.

    Exponential starting slope is at the FIRST POST-MIXING FITTED POINT,
    not at t=0 before mixing. The ordinary early slope is not automatically v0.
    """
    if method not in METHODS:
        raise DataError(f'Unknown analysis method: {method}')
    base = dict(method=method, v0_uM_s=None, v0_se_uM_s=None, k_obs_s_1=None,
                reference_time_s=None, basis='', warning='',
                apparent_fraction_initial_window=None, eligible_for_primary=False)
    if method == EXCLUDE:
        base.update(basis='Excluded by researcher', warning='This trace will not contribute to primary v0.')
        return base
    if fits.get('models') is None:
        raise DataError('Time-course model fit unavailable; correct trace settings before accepting.')
    models = fits['models']
    t = np.asarray(t, dtype=float); a = np.asarray(a, dtype=float)
    retained = t[t >= cutoff_s]
    if len(retained) == 0:
        raise DataError('No data remain after mixing cutoff.')
    t_first = float(retained[0])
    if method == SINGLE:
        if single_start_s < cutoff_s or single_start_s >= single_end_s:
            raise DataError('Single linear window must start at or after mixing cutoff and have positive duration.')
        line = fit_line(t, a, single_start_s, single_end_s)
        v = signal_slope_to_velocity(line['slope_A_s'], extinction, path_cm, blank_slope)
        se = 1000*line['slope_se_A_s']/(extinction*path_cm)
        frac = apparent_postmix_fraction(t, a, t_first, line['end_s'], extinction,
                                         path_cm, initial_nadh)
        base.update(v0_uM_s=v, v0_se_uM_s=se,
                    basis=f'Single linear slope, {line["start_s"]:g}–{line["end_s"]:g} s',
                    reference_time_s=line['start_s'],
                    apparent_fraction_initial_window=frac,
                    eligible_for_primary=True, single_fit=line)
    elif method == EXPONENTIAL:
        trans = models['transition']
        v = signal_slope_to_velocity(trans['m_initial_A_s'], extinction, path_cm, blank_slope)
        interpretable = bool(outcome == 'Detectable time-dependent change' and
                             not trans.get('k_at_bound', False) and trans.get('success', True))
        base.update(v0_uM_s=v,
                    k_obs_s_1=trans['k_obs_s_1'] if interpretable else None,
                    basis='Model-extrapolated initial slope at first retained postmixing time',
                    reference_time_s=t_first, eligible_for_primary=interpretable,
                    warning=('Postmix model initial velocity is NOT a measured t=0 velocity; '
                             'rate uncertainty is not calculated. '
                             + ('' if interpretable else 'Exponential transition is unresolved; do not use its k_obs or v0 for the primary fit.')))
    elif method == EARLY:
        early = fits.get('early')
        if early is None:
            raise DataError('No valid early linear window available.')
        v = signal_slope_to_velocity(early['slope_A_s'], extinction, path_cm, blank_slope)
        se = 1000*early['slope_se_A_s']/(extinction*path_cm)
        frac = apparent_postmix_fraction(t, a, t_first, early['end_s'], extinction,
                                         path_cm, initial_nadh)
        base.update(v0_uM_s=v, v0_se_uM_s=se,
                    basis=f'Early linear slope, {early["start_s"]:g}–{early["end_s"]:g} s',
                    reference_time_s=early['start_s'],
                    apparent_fraction_initial_window=frac,
                    eligible_for_primary=True,
                    warning='Provisional early slope; an unresolved transition may already have occurred during mixing or this window.')
    if base['v0_uM_s'] is not None and (not math.isfinite(base['v0_uM_s']) or base['v0_uM_s'] <= 0):
        base['eligible_for_primary'] = False
        base['warning'] += ' Nonpositive or invalid velocity; cannot be used as primary v0.'
    frac = base['apparent_fraction_initial_window']
    if frac is not None and math.isfinite(frac) and frac > 0.10:
        base['warning'] += (' Apparent postmix absorbance change exceeds 10% NADH equivalents '
                            'during this window; verify actual conversion with appropriate blanks.')
    return base


def apply_approvals(summary, approvals: dict, expected_fingerprints: dict):
    """Append accepted-method columns to a copy of the results table.

    Approval entries are ignored when any fitting input changed. The primary
    dataset contains ONLY still-current researcher-selected kinetic rates.
    """
    result = summary.copy()
    records = []
    for _, row in result.iterrows():
        name = row['file']
        saved = approvals.get(name)
        good = bool(saved and saved.get('fingerprint') == expected_fingerprints.get(name))
        status = ('Accepted' if good else 'Reaccept: settings changed' if saved else 'Not accepted')
        candidate = saved['candidate'] if good else {}
        is_primary = bool(good and saved.get('primary_verified') and
                          candidate.get('eligible_for_primary') and
                          candidate.get('method') != EXCLUDE)
        records.append(dict(review_decision=saved.get('decision', '') if good else '',
                            acceptance_status=status,
                            accepted_method=candidate.get('method', ''),
                            accepted_v0_uM_s=candidate.get('v0_uM_s'),
                            accepted_v0_se_uM_s=candidate.get('v0_se_uM_s'),
                            accepted_k_obs_s_1=candidate.get('k_obs_s_1'),
                            accepted_reference_time_s=candidate.get('reference_time_s'),
                            accepted_initial_window_fraction=candidate.get('apparent_fraction_initial_window'),
                            accepted_basis=candidate.get('basis', ''),
                            accepted_caution=candidate.get('warning', ''),
                            primary_verified=is_primary,
                            reviewer_note=saved.get('note', '') if good else ''))
    import pandas as pd
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def invalidate_stale_review(session_state, signatures: dict,
                            name: str, uid: str, new_signature: str) -> bool:
    """Clear an old kinetic/descriptive choice when its scientific inputs change.

    Return True when an existing decision was invalidated. This runs before
    the selectbox for that trace is instantiated, so Streamlit widget-state
    rules permit assigning NOT_REVIEWED.
    """
    previous = signatures.get(name)
    changed = previous is not None and previous != new_signature
    if changed:
        session_state[f'approval_decision_{uid}'] = NOT_REVIEWED
    signatures[name] = new_signature
    return changed


def preserve_trace_controls(session_state, uids):
    """Keep per-file values when Streamlit stops rendering a file's widgets.

    Streamlit prunes widget keys after users switch files. A separate dictionary
    owned by the session retains the last known value; each rerun synchronizes
    before any new widget for that file is constructed. This prevents accidental
    loss of fitting windows or reviewer notes while navigating among traces.
    """
    preserved = session_state.setdefault('_trace_controls_persistent', {})
    fields = ('mixing_cutoff', 'early_start', 'early_end', 'late_start', 'late_end',
              'single_start', 'single_end')
    for uid in uids:
        entry = preserved.setdefault(uid, {})
        keys = [f'{uid}_{field}' for field in fields]
        keys.extend(f'{prefix}_{uid}' for prefix in
                    ('approval_method', 'approval_primary', 'approval_note', 'approval_decision'))
        for key in keys:
            if key in session_state:
                entry[key] = session_state[key]
            elif key in entry:
                session_state[key] = entry[key]
