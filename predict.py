"""Offline inference adapter over unchanged, frozen statistical research code."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from vendor import mature_decay_core as decay
from vendor import vt_eb_core as vt

MODELS = ('MatureDecay-prior', 'Common-v3-majority600', 'VT-EB')
ASSETS = ('BTCUSDT', 'ETHUSDT', 'SPY', 'QQQ')


def predict_prefix(frame, asset, horizon, as_of, model='MatureDecay-prior'):
    """Consume a completed-bar prefix; the final row is the signal bar s.

    A horizon is h subsequent rows, with strictly earlier training endpoints
    t+h<s. No future target rows, calendar guesses, or API calls are needed.
    """
    if asset not in ASSETS or horizon not in (5, 20) or model not in MODELS:
        raise ValueError('Unsupported asset, horizon, or model')
    if not {'date', 'available', 'close'} <= set(frame.columns):
        raise ValueError('CSV requires date, available, close columns')
    df = frame.loc[:, ['date', 'available', 'close']].copy().reset_index(drop=True)
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError('--as-of must have an explicit timezone')
    if len(df) < 400:
        raise ValueError('At least 400 completed bars are required')
    dates = pd.to_datetime(df.date, errors='raise')
    if dates.isna().any() or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be present, unique, and strictly increasing')
    parsed = [pd.Timestamp(value) for value in df.available]
    if any(pd.isna(value) or value.tzinfo is None for value in parsed):
        raise ValueError('Every available timestamp must have an explicit timezone')
    times = pd.DatetimeIndex(parsed).tz_convert('UTC')
    if not times.is_unique or not times.is_monotonic_increasing or (times > cutoff).any():
        raise ValueError('Supply an ordered, completed prefix with available <= as-of')
    close = pd.to_numeric(df.close, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError('Close prices must be finite and strictly positive')
    df['close'] = close
    df['available'] = [value.isoformat() for value in times]
    s = len(df) - 1
    if model == 'VT-EB':
        config = json.loads((Path(__file__).parent / 'model-config.json').read_text(encoding='utf-8'))
        indices, _ = vt.training_indices(df, s, horizon, {'commonParameters': config['VT-EB']})
        features = vt.features(df)
        probabilities, _ = vt.probability_family(
            features[indices], (close[indices + horizon] > close[indices]).astype(int),
            features[s], horizon, ['VT-EB', 'VTP-EB', 'Kernel-EB', 'Regime-RDA'])
        probability = probabilities['VT-EB']
    else:
        indices = decay.train_indices(close, s, horizon)
        if len(indices) < 250:
            raise ValueError('Fewer than 250 non-flat, strictly matured training labels')
        probability = (decay.mature_prior(close, s, horizon) if model == 'MatureDecay-prior'
                       else float((close[indices + horizon] > close[indices]).mean()))
    if not np.all(indices + horizon < s) or not (times[indices + horizon] < times[s]).all():
        raise ValueError('Training label maturity violates the signal boundary')
    if probability is None or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid probability; no silent fallback')
    return {'model': model, 'asset': asset, 'horizonBars': horizon,
            'signalAvailableAtUTC': times[s].isoformat(), 'asOfUTC': cutoff.tz_convert('UTC').isoformat(),
            'trainedThrough': times[indices[-1] + horizon].isoformat(),
            'trainingRows': len(indices), 'pUp': float(probability),
            'predictedDirection': 1 if probability >= .5 else -1,
            'status': 'research_only_not_calibrated_win_probability'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--asset', choices=ASSETS, required=True)
    parser.add_argument('--horizon', type=int, choices=(5, 20), required=True)
    parser.add_argument('--as-of', required=True, help='Explicit timezone, e.g. 2026-10-24T00:00:00Z')
    parser.add_argument('--model', choices=MODELS, default='MatureDecay-prior')
    args = parser.parse_args()
    result = predict_prefix(pd.read_csv(args.csv), args.asset, args.horizon, args.as_of, args.model)
    print(json.dumps(result, indent=2, allow_nan=False))
