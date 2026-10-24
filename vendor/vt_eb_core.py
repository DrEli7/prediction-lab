"""GPT financial research core: frozen causal state-shrinkage family.

Only --stage dev2025 is implemented. Historical comparison requires a later,
explicit coordinator release and a separate command; it cannot happen here.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    assert path.parent.resolve() == HERE, 'All artifacts must remain in statistical/'
    assert not path.exists(), f'Do not overwrite immutable artifact: {path.name}'
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_inputs():
    spec = json.loads((HERE / 'spec.json').read_text(encoding='utf-8-sig'))
    assert spec['stage'] == 'dev2025'
    assert sha(ROOT / spec['inputSpec']['path']) == spec['inputSpec']['sha256']
    old = json.loads((ROOT / spec['inputSpec']['path']).read_text(encoding='utf-8-sig'))
    assert spec['devOrigins'] == old['devOrigins'] and len(set(spec['devOrigins'])) == 64
    frames = {}
    cap = pd.Timestamp('2026-01-01T00:00:00Z')
    for asset, info in spec['inputs'].items():
        path = ROOT / info['path']
        assert sha(path) == info['sha256'], f'{asset} frozen prices changed'
        df = pd.read_csv(path)
        df = df.loc[pd.to_datetime(df.available, utc=True) < cap].reset_index(drop=True)
        assert df.date.is_unique and df.date.is_monotonic_increasing
        ts = pd.to_datetime(df.available, utc=True)
        assert ts.is_monotonic_increasing and ts.is_unique
        assert np.isfinite(df[['open', 'high', 'low', 'close', 'volume', 'amount']]).all().all()
        assert (df.close > 0).all()
        assert all(d in set(df.date) for d in spec['devOrigins'])
        frames[asset] = df
    return spec, frames


def features(df):
    c = df.close.astype(float)
    lr = np.log(c).diff()
    v20 = lr.rolling(20, min_periods=20).std(ddof=1)
    v60 = lr.rolling(60, min_periods=60).std(ddof=1)
    return np.column_stack([
        np.log(c).diff(5) / (v20 * np.sqrt(5)),
        np.log(c).diff(20) / (v20 * np.sqrt(20)),
        np.log(c).diff(60) / (v60 * np.sqrt(60)),
        np.log(v20 / v60),
    ])


def training_indices(df, signal_index, horizon, spec):
    end = signal_index - horizon
    before = np.arange(max(60, end - spec['commonParameters']['maxTrainingOriginalBars']), end)
    close = df.close.to_numpy(float)
    t = before[close[before + horizon] != close[before]]
    assert len(t) >= spec['commonParameters']['minimumTrainRows']
    assert np.all(t + horizon < signal_index)
    times = pd.to_datetime(df.available, utc=True).astype('int64').to_numpy()
    assert np.all(times[t + horizon] < times[signal_index])
    return t, int(len(before) - len(t))


def smooth_frequency(y, weights, mask, prior, mass=8.):
    mass_actual = float(weights[mask].sum())
    return float((np.dot(weights[mask], y[mask]) + mass * prior) / (mass_actual + mass))


def pullback(z):
    return np.where(z < -.5, 0, np.where(z > .5, 2, 1))


def robust_transform(train, point):
    median = np.median(train, axis=0)
    q = np.quantile(train, [.25, .75], axis=0)
    scale = np.maximum(q[1] - q[0], 1e-6)
    return np.clip((train - median) / scale, -4., 4.), np.clip((point - median) / scale, -4., 4.)


def probability_family(xtrain, y, point, horizon, names):
    """No labels or features later than this snapshot are accepted by this API."""
    assert xtrain.ndim == 2 and xtrain.shape == (len(y), 4) and point.shape == (4,)
    assert np.isfinite(xtrain).all() and np.isfinite(point).all()
    assert set(np.unique(y)) <= {0, 1} and len(y) > 0
    weights = np.full(len(y), 1. / horizon)
    total = float(weights.sum())
    p0 = float((np.dot(weights, y) + 1.) / (total + 2.))
    cut = float(np.median(xtrain[:, 3]))
    volatility = xtrain[:, 3] >= cut
    v_now = bool(point[3] >= cut)
    trend = xtrain[:, 2] >= 0
    trend_now = bool(point[2] >= 0)
    same_v = volatility == v_now
    same_vt = same_v & (trend == trend_now)
    pv = smooth_frequency(y, weights, same_v, p0)
    pvt = smooth_frequency(y, weights, same_vt, pv)
    same_vtp = same_vt & (pullback(xtrain[:, 0]) == int(pullback(point[0])))
    pvtp = smooth_frequency(y, weights, same_vtp, pvt)
    x, current = robust_transform(xtrain, point)
    kernel = np.exp(-.5 * np.sum((x - current) ** 2, axis=1))
    pk = float((np.dot(weights * kernel, y) + 8. * p0) / (np.dot(weights, kernel) + 8.))
    mean = np.average(x, axis=0, weights=weights)
    mu_global = []
    mu_state = []
    for label in (0, 1):
        same_c = y == label
        class_mass = float(weights[same_c].sum())
        m = (np.sum(weights[same_c, None] * x[same_c], axis=0) + 8. * mean) / (class_mass + 8.)
        mu_global.append(m)
        same_sc = same_v & same_c
        state_class_mass = float(weights[same_sc].sum())
        ms = (np.sum(weights[same_sc, None] * x[same_sc], axis=0) + 8. * m) / (state_class_mass + 8.)
        mu_state.append(ms)
    mu_global = np.array(mu_global)
    mu_state = np.array(mu_state)
    variance_global = np.maximum(np.average((x - mu_global[y]) ** 2, weights=weights, axis=0), .1)
    state_mass = float(weights[same_v].sum())
    state_residual_sum = np.sum(weights[same_v, None] * (x[same_v] - mu_state[y[same_v]]) ** 2, axis=0)
    variance_state = np.maximum((state_residual_sum + 16. * variance_global) / (state_mass + 16.), .1)
    logit = np.log(pv / (1. - pv)) + np.sum(
        ((current - mu_state[0]) ** 2 - (current - mu_state[1]) ** 2) / (2. * variance_state))
    prda = float(expit(np.clip(logit, -35., 35.)))
    probs = {'VT-EB': pvt, 'VTP-EB': pvtp, 'Kernel-EB': pk, 'Regime-RDA': prda}
    assert set(probs) == set(names)
    assert all(np.isfinite(p) and 0 <= p <= 1 for p in probs.values())
    return probs, {
        'rawTrainingRows': len(y), 'effectiveWeightMass': total, 'globalShrunkFrequency': p0,
        'volatilityThreshold': cut, 'volatilityState': int(v_now), 'trendState': int(trend_now),
        'pullbackState': int(pullback(point[0])),
        'volatilityEffectiveMass': state_mass,
        'jointEffectiveMass': float(weights[same_vt].sum()),
        'pullbackEffectiveMass': float(weights[same_vtp].sum()),
        'kernelEffectiveMass': float(np.dot(weights, kernel)),
        'volatilityParentProbability': pv,
    }


def snapshot(df, x, origin_date, horizon, spec):
    locations = np.flatnonzero(df.date.to_numpy() == origin_date)
    assert len(locations) == 1
    i = int(locations[0])
    s = i - 1
    t, flats = training_indices(df, s, horizon, spec)
    assert np.isfinite(x[np.r_[t, s]]).all()
    close = df.close.to_numpy(float)
    y = (close[t + horizon] > close[t]).astype(int)
    frequency = float(y.mean())
    failures = {}
    try:
        probs, state = probability_family(x[t], y, x[s], horizon, spec['candidateOrder'])
    except Exception as exc:
        # Never silently remove rows, even if a model fails. Baseline is available
        # after the independent temporal selection has succeeded.
        probs = {n: frequency for n in spec['candidateOrder']}
        failures = {n: f'{type(exc).__name__}: {exc}' for n in spec['candidateOrder']}
        state = {'exceptionTrace': traceback.format_exc()}
    trained = str(df.available.iloc[t[-1] + horizon])
    signal = str(df.available.iloc[s])
    assert pd.Timestamp(trained) < pd.Timestamp(signal)
    state.update({'originDate': origin_date, 'horizon': horizon, 'trainRows': len(t),
                  'excludedFlatTrainingLabels': flats, 'trainingFirstFeature': str(df.available.iloc[t[0]]),
                  'trainingLastFeature': str(df.available.iloc[t[-1]]),
                  'trainedThrough': trained, 'featureAvailableThrough': signal,
                  'signalIndex': s, 'trainLastIndex': int(t[-1]),
                  'trainLastLabelIndex': int(t[-1] + horizon)})
    probs.update({'Training-majority': frequency, 'Always-up': 1., 'Always-down': 0.})
    records = []
    for candidate, p in probs.items():
        records.append({'originDate': origin_date, 'horizon': horizon, 'candidateId': candidate,
                        'predictedDirection': 1 if p >= .5 else -1, 'pUp': p,
                        'trainedThrough': trained, 'featureAvailableThrough': signal,
                        'signalAvailableAtUTC': signal, 'contextLastBarDate': str(df.date.iloc[s]),
                        'previous': float(close[s]), 'targetDate': str(df.date.iloc[i + horizon - 1]),
                        'targetEndUTC': str(df.available.iloc[i + horizon - 1]),
                        'status': 'fallback' if candidate in failures else 'ok',
                        'failureReason': failures.get(candidate),
                        'fallback': 'Training-majority' if candidate in failures else None})
    return records, state


def status(stage, progress, **extra):
    path = HERE / 'status.json'
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    obj.update({'stage': stage, 'progress': progress, 'updatedAt': datetime.now(timezone.utc).isoformat(), **extra})
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verify_source_manifest():
    manifest = json.loads((HERE / 'source-manifest.json').read_text())
    assert manifest['specSHA256'] == sha(HERE / 'spec.json')
    for name, expected in manifest['sources'].items():
        assert sha(HERE / name) == expected, f'Frozen source changed: {name}'
    return manifest


def run():
    assert not (HERE / 'dev-predictions.json').exists(), 'Never overwrite development predictions'
    verify_source_manifest()
    spec, frames = load_inputs()
    rows, audit = [], []
    began = time.perf_counter()
    for asset, df in frames.items():
        x = features(df)
        for origin_index, origin in enumerate(spec['devOrigins']):
            for horizon in spec['horizons']:
                records, state = snapshot(df, x, origin, horizon, spec)
                rows.extend([dict(r, asset=asset, originIndex=origin_index, fold=origin_index // 16) for r in records])
                audit.append(dict(state, asset=asset, originIndex=origin_index))
        print(json.dumps({'asset': asset, 'snapshotCount': len(audit), 'elapsedSeconds': time.perf_counter() - began}), flush=True)
        status('running_dev2025', f'{asset} completed; {len(audit)}/512 asset/origin/horizon snapshots')
    order = spec['candidateOrder'] + ['Training-majority', 'Always-up', 'Always-down']
    rows.sort(key=lambda r: (order.index(r['candidateId']), r['originIndex'], spec['assets'].index(r['asset']), r['horizon']))
    assert len(rows) == 512 * len(order) and len(audit) == 512
    result = {'stage': 'dev2025', 'createdAtUTC': datetime.now(timezone.utc).isoformat(),
              'specSHA256': sha(HERE / 'spec.json'), 'codeSHA256': sha(HERE / 'study.py'),
              'sourceManifestSHA256': sha(HERE / 'source-manifest.json'),
              'rows': rows, 'audit': audit,
              'new2026ScoresComputed': False, 'elapsedSeconds': time.perf_counter() - began}
    write_json(HERE / 'dev-predictions.json', result)
    status('dev_predictions_complete', 'All frozen 2025 candidates predicted; scoring/audit next')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['dev2025'], required=True)
    args = parser.parse_args()
    run()
