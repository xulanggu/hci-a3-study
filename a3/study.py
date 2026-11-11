"""Pure study logic, independently testable without Streamlit."""
import csv
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from config import ROOT, LABELS, N_MAIN, N_AI_WRONG, N_ATTENTION


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize_id(value):
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,20}", value):
        raise ValueError("Use 3–20 letters, numbers, or underscores.")
    return value


def load_data():
    def read(name):
        with (ROOT / 'data' / name).open(newline='') as f:
            return list(csv.DictReader(f))
    study, dev, manifest = read('study_predictions.csv'), read('dev_manifest.csv'), read('study_manifest.csv')
    assert len(study) == 240 and len(dev) == 40, 'Expected 240 study and 40 development images'
    assert len({r['trial_id'] for r in study + dev}) == 280
    assert len({r['image_sha256'] for r in study + dev}) == 280
    assert {r['trial_id']: {k:r[k] for k in manifest[0]} for r in study} == {r['trial_id']:r for r in manifest}
    for row in study + dev:
        assert float(row['agreement']) >= .6
        assert row['reference_label'] in LABELS
        row['image_id'] = row['trial_id']
        if 'model_prediction' in row:
            row['ai_correct_frozen'] = row['model_prediction'] == row['reference_label']
            row['probs'] = json.loads(row['model_scores'])
            assert int(row['correct']) == int(row['ai_correct_frozen'])
            assert len(row['probs']) == 8 and abs(sum(row['probs']) - 1) < 1e-6
            order = sorted(range(8), key=lambda i: -row['probs'][i])
            assert [row['model_prediction'], row['model_second_choice']] == [LABELS[i] for i in order[:2]]
    correct = sum(r['ai_correct_frozen'] for r in study)
    assert correct == 187, f'Frozen accuracy differs: {correct}/240; expected 187/240 (53 wrong)'
    return study, dev


def sample_plan(study, dev, seen, seed, n_main=N_MAIN, n_wrong=N_AI_WRONG, n_attention=N_ATTENTION):
    rng = random.Random(seed)
    warnings, main = [], []
    for correct, count in [(True, n_main-n_wrong), (False, n_wrong)]:
        pool = [r for r in study if r['image_id'] not in seen and r['ai_correct_frozen'] == correct]
        if len(pool) < count:
            warnings.append(f"Insufficient {'correct' if correct else 'wrong'} stratum: requested {count}, available {len(pool)}.")
        main.extend({'image_id': r['image_id'], 'is_attention_check': False, 'attention_target': None}
                    for r in rng.sample(pool, min(count, len(pool))))
    rng.shuffle(main)
    dev_pool = [r for r in dev if r['image_id'] not in seen]
    # Need three main trials before the first check, and a main trial between checks.
    count = min(n_attention, len(dev_pool), max(0, len(main)-2))
    if count < n_attention:
        warnings.append(f'Only {count} attention checks possible with the remaining pool.')
    # Uniform nonadjacent positions using the compressed-combination mapping.
    positions = {p+i for i, p in enumerate(sorted(rng.sample(range(3, len(main)+1), count)))}
    checks = iter(rng.sample(dev_pool, count))
    mains = iter(main)
    plan = []
    for i in range(len(main)+count):
        if i in positions:
            plan.append({'image_id': next(checks)['image_id'], 'is_attention_check': True,
                         'attention_target': rng.choice(LABELS)})
        else:
            plan.append(next(mains))
    return plan, warnings


def derived(final_label, true_label, ai=None, initial_label=None, attention_target=None):
    if attention_target is not None:
        return dict(correct=None, switched=None, followed_ai_top1=None, chose_ai_top2=None,
                    attention_passed=final_label == attention_target)
    return dict(correct=final_label == true_label,
                switched=(initial_label != final_label) if initial_label is not None else None,
                followed_ai_top1=(final_label == ai['ai_top1']) if ai else None,
                chose_ai_top2=(final_label == ai['ai_top2']) if ai else None,
                attention_passed=None)


def model_status():
    expected = json.loads((ROOT/'data/a2_study_run.json').read_text())['model_sha256']
    try:
        actual = hashlib.sha256((ROOT/'models/emotion-ferplus-8.onnx').read_bytes()).hexdigest()
        return dict(expected=expected, actual=actual, matches=actual == expected)
    except OSError:
        return dict(expected=expected, actual=None, matches=False)
