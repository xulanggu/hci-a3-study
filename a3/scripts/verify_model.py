"""Verify all 240 frozen predictions and all 280 pixel hashes. No fallback allowed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
from PIL import Image
from config import ROOT, LABELS
from study import load_data, model_status
from vendor.a2_predictor import Predictor


def main():
    study, dev = load_data()
    status = model_status()
    assert status['matches'], f'Checkpoint mismatch: {status}'
    for row in study+dev:
        with Image.open(ROOT/row['image_path']) as im:
            assert im.size == (48,48)
            assert hashlib.sha256(im.convert('L').tobytes()).hexdigest() == row['image_sha256'], row['image_id']
    net = Predictor(ROOT/'models/emotion-ferplus-8.onnx')
    mismatches, correct = [], []
    for row in study:
        first, second, _ = net.predict(ROOT/row['image_path'])
        if (first,second) != (row['model_prediction'], row['model_second_choice']):
            mismatches.append(row['image_id'])
        correct.append(first == row['reference_label'])
    report = dict(top1_top2_match=f'{len(study)-len(mismatches)}/240', accuracy=sum(correct)/240,
                  correct=sum(correct), wrong=240-sum(correct), checkpoint=status,
                  per_class_recall={label:sum(c for r,c in zip(study,correct) if r['reference_label']==label)/sum(r['reference_label']==label for r in study) for label in LABELS},
                  mismatches=mismatches)
    print(json.dumps(report, indent=2))
    assert not mismatches, 'Frozen top-1/top-2 mismatch'
    assert sum(correct)==187, 'Expected 187 correct / 53 wrong'


if __name__=='__main__':
    main()
