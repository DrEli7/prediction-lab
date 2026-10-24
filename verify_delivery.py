"""Verify release hashes and already-frozen outputs; no inference or training."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / 'release-manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        data = path.read_bytes()
        assert len(data) == item['bytes'], item['path']
        assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
    with (ROOT / 'results/frozen_2025_predictions.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = {'MatureDecay-prior': 303, 'Common-v3-majority600': 304, 'VT-EB': 301}
    metrics = {}
    keysets = []
    truths = {}
    for model, expected_hits in expected.items():
        selected = [row for row in rows if row['model'] == model]
        keys = [(row['asset'], row['originDate'], int(row['horizon'])) for row in selected]
        assert len(selected) == len(set(keys)) == 512
        keysets.append(set(keys))
        assert len({row['originDate'] for row in selected}) == 64
        counts = Counter((row['asset'], int(row['horizon'])) for row in selected)
        assert set(counts) == {(a, h) for a in ('BTCUSDT', 'ETHUSDT', 'SPY', 'QQQ') for h in (5, 20)}
        assert all(value == 64 for value in counts.values())
        hits = 0
        errors = []
        for row in selected:
            key = (row['asset'], row['originDate'], int(row['horizon']))
            actual = int(row['actualDirection'])
            predicted = int(row['predictedDirection'])
            probability = float(row['pUp'])
            assert actual in (-1, 1) and predicted in (-1, 1) and math.isfinite(probability)
            assert 0 <= probability <= 1 and predicted == (1 if probability >= .5 else -1)
            assert row['status'] == 'ok'
            assert datetime.fromisoformat(row['trainedThrough']) < datetime.fromisoformat(row['signalAvailableAtUTC'])
            assert key not in truths or truths[key] == actual
            truths[key] = actual
            hits += actual == predicted
            errors.append((probability - (actual == 1)) ** 2)
        assert hits == expected_hits, (model, hits)
        metrics[model] = {'hits': hits, 'n': 512, 'accuracy': hits / 512,
                          'brier': math.fsum(errors) / 512}
    assert keysets[0] == keysets[1] == keysets[2]
    assert len(rows) == 1536
    for source in ROOT.rglob('*.py'):
        compile(source.read_text(encoding='utf-8-sig'), str(source), 'exec')
    return {'verified': True, 'fileCount': len(manifest['files']),
            'frozenPredictionRows': len(rows), 'metrics': metrics,
            'newInferenceOrTrainingPerformed': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
