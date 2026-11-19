"""MuSiQue-Ans benchmark starter. Python 3.10+, standard library only.
Official reference: https://github.com/stonybrooknlp/musique
Run from your project root. No model inference is implemented in this starter.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import random
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hops(row):
    return len(row['question_decomposition'])


def support(row):
    return {p['idx'] for p in row['paragraphs'] if p['is_supporting']}


def validate(row):
    assert row['answerable'] is True, f"Not MuSiQue-Ans: {row['id']}"
    assert hops(row) in (2, 3, 4), row['id']
    ps = row['paragraphs']
    ids = {p['idx'] for p in ps}
    # Some MuSiQue contexts contain fewer than 20 paragraphs.
    assert 0 < len(ps) <= 20, f"Expected 1-20 paragraphs, got {len(ps)}: {row['id']}"
    assert len(ps) == len(ids), f"Duplicate paragraph IDs: {row['id']}"
    assert support(row) and all(s['paragraph_support_idx'] in support(row) for s in row['question_decomposition']), row['id']
    assert isinstance(row['answer'], str) and all(isinstance(a, str) for a in row['answer_aliases']), row['id']


def avg(x):
    return statistics.fmean(x) if x else None


def analyze(path):
    counts = Counter(); ids = set(); groups = defaultdict(list)
    for row in read_jsonl(path):
        validate(row)
        assert row['id'] not in ids, f"Duplicate ID: {row['id']}"
        ids.add(row['id']); counts[hops(row)] += 1
        groups[hops(row)].append({
            'context_words': sum(len(p['paragraph_text'].split()) for p in row['paragraphs']),
            'question_words': len(row['question'].split()),
            'support_paragraphs': len(support(row)),
            'answer_aliases': len(row['answer_aliases']),
        })
    print(json.dumps({'file': str(path), 'sha256': sha256(path), 'n': len(ids), 'by_hop': {
        str(h): {'count': counts[h], **{f'mean_{k}': avg([r[k] for r in rows]) for k in rows[0]},
                 'median_context_words': statistics.median(r['context_words'] for r in rows)}
        for h, rows in sorted(groups.items())}}, indent=2))


def inputs(row):
    return {'id': row['id'], 'question': row['question'], 'paragraphs': [
        {k: p[k] for k in ('idx', 'title', 'paragraph_text')} for p in row['paragraphs']]}


def save_split(out, name, rows):
    rows = list(rows)
    write_jsonl(out / name / 'inputs.jsonl', map(inputs, rows))
    write_jsonl(out / name / 'gold.jsonl', rows)
    return {'count': len(rows), 'by_hop': dict(sorted(Counter(map(hops, rows)).items())),
            'ids': [r['id'] for r in rows]}


def prepare(train, dev, out, seed, smoke_n, tune_n):
    if min(smoke_n, tune_n) < 1:
        raise ValueError('Sample counts must be positive')
    # Reservoir sampling: avoids holding all 241 MB of train data in memory.
    rng = random.Random(seed); pools = defaultdict(list); counts = Counter(); seen = set()
    n = smoke_n + tune_n
    for row in read_jsonl(train):
        validate(row); rid = row['id']; h = hops(row)
        if rid in seen: raise ValueError(f'Duplicate train ID: {rid}')
        seen.add(rid); counts[h] += 1; pool = pools[h]
        if len(pool) < n: pool.append(row)
        else:
            i = rng.randrange(counts[h])
            if i < n: pool[i] = row
    smoke, tune = [], []
    for h in (2, 3, 4):
        if len(pools[h]) != n: raise ValueError(f'Not enough train examples for {h}-hop')
        rng.shuffle(pools[h]); smoke.extend(pools[h][:smoke_n]); tune.extend(pools[h][smoke_n:])
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    def dev_rows():
        dev_seen = set()
        for row in read_jsonl(dev):
            validate(row); rid = row['id']
            if rid in seen or rid in dev_seen: raise ValueError(f'Overlapping or duplicate ID: {rid}')
            dev_seen.add(rid); yield row
    manifest = {'seed': seed, 'source_sha256': {'train': sha256(train), 'dev': sha256(dev)}, 'splits': {
        'smoke': save_split(out, 'smoke', smoke),
        'calibration': save_split(out, 'calibration', tune),
        'evaluation': save_split(out, 'evaluation', dev_rows())}}
    dump(out/'manifest.json', manifest)
    print(json.dumps({k: {a: b for a,b in v.items() if a!='ids'} for k,v in manifest['splits'].items()}, indent=2))


def normalize(text):
    # Standard SQuAD-style normalization used by MuSiQue.
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_scores(pred, refs):
    p = normalize(pred)
    def f1(ref):
        r, t = normalize(ref).split(), p.split()
        if not r or not t: return float(r == t)
        common = sum((Counter(r) & Counter(t)).values())
        return 2 * common / (len(r) + len(t))
    return max(float(p == normalize(r)) for r in refs), max(f1(r) for r in refs)


def evaluate(gold_path, pred_path, out):
    predictions = {}
    for p in read_jsonl(pred_path):
        if p['id'] in predictions: raise ValueError(f"Duplicate prediction {p['id']}")
        predictions[p['id']] = p
    records = []; official = []; gold_ids = set()
    for row in read_jsonl(gold_path):
        rid = row['id']; validate(row)
        if rid in gold_ids: raise ValueError(f'Duplicate gold ID {rid}')
        gold_ids.add(rid)
        if rid not in predictions: raise ValueError(f'Missing prediction {rid}')
        p = predictions[rid]
        if not isinstance(p['predicted_answer'], str): raise ValueError(f'Invalid answer {rid}')
        if not isinstance(p['predicted_answerable'], bool): raise ValueError(f'Invalid answerability {rid}')
        selected = set(p['predicted_support_idxs']); truth = support(row)
        if not selected <= {x['idx'] for x in row['paragraphs']}: raise ValueError(f'Invalid paragraph ID {rid}')
        em, f1 = answer_scores(p['predicted_answer'], [row['answer'], *row['answer_aliases']])
        common = len(selected & truth)
        r = {'id': rid, 'hop': hops(row), 'answer_em': em, 'answer_f1': f1,
             'support_precision': common / len(selected) if selected else 0.,
             'support_recall': common / len(truth), 'support_f1': 2*common/(len(selected)+len(truth)),
             'all_support_recovered': float(truth <= selected), 'support_em': float(truth == selected)}
        for key in ('raw_context_tokens', 'compressed_context_tokens', 'controller_input_tokens', 'controller_output_tokens', 'reader_input_tokens', 'reader_output_tokens', 'compression_seconds', 'reader_seconds', 'total_seconds', 'iterations', 'llm_calls'):
            val = p.get(key)
            if val is not None:
                if not isinstance(val, (float,int)) or isinstance(val,bool) or not math.isfinite(val) or val < 0: raise ValueError(f'Invalid telemetry {rid}: {key}')
            r[key] = val
        raw, compressed = r['raw_context_tokens'], r['compressed_context_tokens']
        r['compression_ratio'] = compressed/raw if raw and compressed is not None else None
        token_fields = [r[x] for x in ('controller_input_tokens','controller_output_tokens','reader_input_tokens','reader_output_tokens')]
        r['total_llm_tokens'] = sum(token_fields) if all(x is not None for x in token_fields) else None
        records.append(r)
        official.append({'id':rid, 'predicted_answer': p['predicted_answer'], 'predicted_support_idxs': sorted(selected), 'predicted_answerable':p['predicted_answerable']})
    if set(predictions) != gold_ids or not records: raise ValueError('Gold and prediction IDs must match exactly and be nonempty')
    def summary(rs):
        fields = [k for k in rs[0] if k not in ('id','hop')]
        paired = [r for r in rs if r['raw_context_tokens'] and r['compressed_context_tokens'] is not None]
        raw_total = sum(r['raw_context_tokens'] for r in paired)
        return {'n': len(rs), 'weighted_compression_ratio': sum(r['compressed_context_tokens'] for r in paired)/raw_total if raw_total else None, 'weighted_compression_n':len(paired), **{k: avg([r[k] for r in rs if r[k] is not None]) for k in fields},
                'metric_counts': {k: sum(r[k] is not None for r in rs) for k in fields}}
    result = {'overall': summary(records), 'by_hop': {str(h):summary([r for r in records if r['hop']==h]) for h in sorted({r['hop'] for r in records})}}
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    dump(out/'summary.json', result)
    with (out/'per_sample.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    write_jsonl(out/'official_predictions.jsonl', official)
    print(json.dumps(result['overall'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze'); a.add_argument('file')
    p = sub.add_parser('prepare'); p.add_argument('--train', required=True); p.add_argument('--dev', required=True); p.add_argument('--out', default='data/benchmark'); p.add_argument('--seed', type=int, default=42); p.add_argument('--smoke-per-hop',type=int,default=10); p.add_argument('--calibration-per-hop',type=int,default=50)
    e = sub.add_parser('evaluate'); e.add_argument('--gold',required=True); e.add_argument('--predictions',required=True); e.add_argument('--out',required=True)
    args = parser.parse_args()
    if args.cmd=='analyze': analyze(args.file)
    elif args.cmd=='prepare': prepare(args.train,args.dev,args.out,args.seed,args.smoke_per_hop,args.calibration_per_hop)
    else: evaluate(args.gold,args.predictions,args.out)

if __name__=='__main__': main()
