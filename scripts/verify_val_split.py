"""Verify validation split has strict temporal ordering and no overlap with test."""
import json
from collections import defaultdict

train = [json.loads(l) for l in open('data/grepo_text/grepo_train_split.jsonl')]
val = [json.loads(l) for l in open('data/grepo_text/grepo_val.jsonl')]
test = [json.loads(l) for l in open('data/grepo_text/grepo_test.jsonl')]

# Check split fields
assert all(d['split'] == 'train' for d in train), "Train split field mismatch"
assert all(d['split'] == 'val' for d in val), "Val split field mismatch"
assert all(d['split'] == 'test' for d in test), "Test split field mismatch"

# Check no issue overlap
train_keys = {(d['repo'], d['issue_id']) for d in train}
val_keys = {(d['repo'], d['issue_id']) for d in val}
test_keys = {(d['repo'], d['issue_id']) for d in test}
assert not (train_keys & val_keys), "Train/val issue overlap!"
assert not (val_keys & test_keys), "Val/test issue overlap!"
assert not (train_keys & test_keys), "Train/test issue overlap!"

# Check temporal ordering (train < val per repo)
train_ts = defaultdict(list)
val_ts = defaultdict(list)
for d in train: train_ts[d['repo']].append(d.get('timestamp', ''))
for d in val: val_ts[d['repo']].append(d.get('timestamp', ''))

overlap = 0
for repo in set(train_ts) & set(val_ts):
    if max(train_ts[repo]) > min(val_ts[repo]):
        overlap += 1

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Train/val temporal overlap: {overlap}/{len(set(train_ts) & set(val_ts))} repos")
assert overlap == 0, f"Train/val temporal overlap in {overlap} repos!"
print("ALL CHECKS PASSED")
