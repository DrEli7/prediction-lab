"""Financial core for batch2; all design choices frozen before selection."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COORD = ROOT / 'research-team/coordinator/batch-2'
CANDIDATES = ['Majority600-reference', 'MatureDecay-prior', 'EqualPrior-conditional', 'DecayPrior-residual']
BASELINES = ['Common-v3-majority600', 'Conditional-window-majority', 'Always-up', 'Always-down']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(name, obj):
    path = HERE / name
    assert path.parent.resolve() == HERE and not path.exists(), 'Immutable artifact or wrong directory'
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load():
    spec = read(HERE / 'spec.json')
    assert sha(COORD / 'origins.json') == spec['originsSHA256'], 'Coordinator origins changed'
    frames = {}
    for asset, info in spec['inputs'].items():
        path = ROOT / info['path']
        assert sha(path) == info['sha256']
        df = pd.read_csv(path)
        df = df.loc[pd.to_datetime(df.available, utc=True) < pd.Timestamp('2026-01-01T00:00:00Z')].reset_index(drop=True)
        assert df.date.is_unique and df.date.is_monotonic_increasing
        assert pd.to_datetime(df.available, utc=True).is_monotonic_increasing
        assert np.isfinite(df.close).all() and (df.close > 0).all()
        frames[asset] = df
    return spec, frames


def raw_features(df):
    c = df.close.astype(float)
    lr = np.log(c).diff()
    v20 = lr.rolling(20).std(ddof=1)
    v60 = lr.rolling(60).std(ddof=1)
    return np.column_stack([np.log(c).diff(5) / (v20 * np.sqrt(5)),
                            np.log(c).diff(20) / (v20 * np.sqrt(20)),
                            np.log(c).diff(60) / (v60 * np.sqrt(60)), np.log(v20 / v60)])


def train_indices(c, s, h):
    end = s - h
    raw = np.arange(max(60, end - 600), end)
    return raw[c[raw + h] != c[raw]]


def decay_weights(t, s, h):
    # Unit weight for a label ending one bar before signal; decay on label age.
    return np.exp2(-(s - (t + h) - 1.) / 252.) / h


def mature_prior(c, s, h, minimum=120):
    t = train_indices(c, s, h)
    if len(t) < minimum:
        return None
    y = (c[t + h] > c[t]).astype(float)
    w = decay_weights(t, s, h)
    return float((np.dot(w, y) + 1.) / (w.sum() + 2.))


def prior_series(c, h):
    # Each element is an independent historical as-of computation. No label
    # endpoint at or after its feature time is included in that element.
    out = np.full(len(c), np.nan)
    for s in range(60 + 120 + h, len(c)):
        p = mature_prior(c, s, h)
        if p is not None:
            out[s] = p
    return out


def basis(train, point):
    median = np.median(train, axis=0)
    q = np.quantile(train, [.25, .75], axis=0)
    scale = np.maximum(q[1] - q[0], 1e-6)
    z = np.clip((train - median) / scale, -3., 3.)
    current = np.clip((point - median) / scale, -3., 3.)
    def expand(x):
        return np.column_stack([x, x[:, 0] * x[:, 2] / 3., x[:, 0] * x[:, 3] / 3.,
                                x[:, 1] * x[:, 3] / 3., x[:, 2] ** 2 / 3.])
    return expand(z), expand(current.reshape(1, 4))[0]


def ridge_component(x, target, point, weights):
    mean = np.average(x, weights=weights, axis=0)
    xc = x - mean
    gram = (xc.T * weights) @ xc + 8. * np.eye(x.shape[1])
    beta = np.linalg.solve(gram, (xc.T * weights) @ target)
    return float((point - mean) @ beta), {'coefficientL2Norm': float(np.linalg.norm(beta)),
                                         'effectiveWeightMass': float(weights.sum())}


def snapshot(df, x, priors, origin, h, names):
    i = df.date.tolist().index(origin)
    s = i - 1
    c = df.close.to_numpy(float)
    t = train_indices(c, s, h)
    assert len(t) >= 250 and np.all(t + h < s)
    times = pd.to_datetime(df.available, utc=True).astype('int64').to_numpy()
    assert np.all(times[t + h] < times[s])
    y = (c[t + h] > c[t]).astype(int)
    p_common = float(y.mean())
    conditional_t = t[np.isfinite(priors[h][t])]
    assert len(conditional_t) >= 250
    assert np.isfinite(x[np.r_[conditional_t, s]]).all()
    yc = (c[conditional_t + h] > c[conditional_t]).astype(int)
    p_conditional_majority = float(yc.mean())
    p_prior = mature_prior(c, s, h)
    assert p_prior is not None
    probs = {'Majority600-reference': p_common, 'MatureDecay-prior': p_prior}
    failures = {}
    details = {}
    if any(n in names for n in ('EqualPrior-conditional', 'DecayPrior-residual')):
        xb, point = basis(x[conditional_t], x[s])
        w = decay_weights(conditional_t, s, h)
        for name in ('EqualPrior-conditional', 'DecayPrior-residual'):
            if name not in names:
                continue
            try:
                if name == 'EqualPrior-conditional':
                    assert set(yc) == {0, 1}, 'Conditional class absent'
                    balanced = w.copy()
                    for cls in (0, 1):
                        mask = yc == cls
                        balanced[mask] *= w.sum() / (2. * w[mask].sum())
                    component, detail = ridge_component(xb, 2. * yc - 1., point, balanced)
                    probability = .5 + .5 * component
                    detail['fixedClassPrior'] = .5
                else:
                    historical_prior = priors[h][conditional_t]
                    component, detail = ridge_component(xb, yc - historical_prior, point, w)
                    probability = p_prior + component
                    detail['currentMaturePrior'] = p_prior
                    detail['historicalPriorMin'] = float(historical_prior.min())
                    detail['historicalPriorMax'] = float(historical_prior.max())
                    detail['historicalPriorBoundary'] = 'At each training feature t, only labels u+h<t are used, max600 and min120 nonflat rows'
                probs[name] = float(np.clip(probability, 1e-6, 1. - 1e-6))
                details[name] = dict(detail, conditionalComponent=component)
            except Exception as exc:
                probs[name] = p_conditional_majority
                failures[name] = f'{type(exc).__name__}: {exc}'
    probs.update({'Common-v3-majority600': p_common, 'Conditional-window-majority': p_conditional_majority,
                  'Always-up': 1., 'Always-down': 0.})
    records = []
    for name in names:
        p = probs[name]
        assert np.isfinite(p) and 0 <= p <= 1
        tt = conditional_t if name in ('EqualPrior-conditional', 'DecayPrior-residual', 'Conditional-window-majority') else t
        records.append({'candidateId': name, 'originDate': origin, 'horizon': h,
                        'predictedDirection': 1 if p >= .5 else -1, 'pUp': p,
                        'trainedThrough': df.available.iloc[tt[-1] + h],
                        'featureAvailableThrough': df.available.iloc[s], 'signalAvailableAtUTC': df.available.iloc[s],
                        'contextLastBarDate': df.date.iloc[s], 'previous': float(c[s]),
                        'targetDate': df.date.iloc[i + h - 1], 'targetEndUTC': df.available.iloc[i + h - 1],
                        'trainingRows': len(tt), 'status': 'fallback' if name in failures else 'ok',
                        'failureReason': failures.get(name),
                        'fallback': 'Conditional-window-majority' if name in failures else None})
    audit = {'originDate': origin, 'horizon': h, 'signalIndex': s, 'commonRows': len(t),
             'conditionalRows': len(conditional_t), 'lastCommonLabelIndex': int(t[-1] + h),
             'lastConditionalLabelIndex': int(conditional_t[-1] + h),
             'firstConditionalFeatureIndex': int(conditional_t[0]),
             'currentMaturePrior': p_prior, 'conditionalDetails': details,
             'strictMaturity': bool(times[t[-1] + h] < times[s] and times[conditional_t[-1] + h] < times[s])}
    return records, audit


def metrics(rows):
    y = np.array([r['actualDirection'] for r in rows])
    pred = np.array([r['predictedDirection'] for r in rows])
    hit = y == pred
    up, down = y == 1, y == -1
    return {'n': len(rows), 'hits': int(hit.sum()), 'accuracy': float(hit.mean()),
            'balancedAccuracy': float((hit[up].mean() + hit[down].mean()) / 2.) if up.any() and down.any() else None,
            'actualUp': int(up.sum()), 'predictedUp': int((pred == 1).sum()),
            'brierScore': float(np.mean((np.array([r['pUp'] for r in rows]) - up) ** 2)),
            'failures': sum(r['status'] != 'ok' for r in rows), 'coverage': 1.}


def score(rows, spec):
    out = metrics(rows)
    cells = [dict(asset=a, horizon=h, **metrics([r for r in rows if r['asset'] == a and r['horizon'] == h]))
             for a in spec['assets'] for h in spec['horizons']]
    values = [c['balancedAccuracy'] for c in cells]
    out.update({'byCell': cells, 'macroBalancedAccuracy': float(np.mean(values)) if all(v is not None for v in values) else None,
                'macroUndefinedCells': [f"{c['asset']}/{c['horizon']}" for c in cells if c['balancedAccuracy'] is None],
                'byAsset': {a: metrics([r for r in rows if r['asset'] == a]) for a in spec['assets']},
                'byHorizon': {str(h): metrics([r for r in rows if r['horizon'] == h]) for h in spec['horizons']}})
    return out


def update_status(stage, progress, **extra):
    obj = read(HERE / 'status.json')
    obj.update(stage=stage, progress=progress, updatedAt=datetime.now(timezone.utc).isoformat(), **extra)
    (HERE / 'status.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
