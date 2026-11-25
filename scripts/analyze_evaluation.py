"""Read-only audit of an exported run. No model imports or inference."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import read_jsonl, answer_scores, normalize, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--gold", default=str(ROOT / "data/benchmark/evaluation/gold.jsonl"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Choose a fresh audit directory")
    gold = {r['id']: r for r in read_jsonl(args.gold)}
    groups = defaultdict(list)
    counts = Counter()
    reasons = Counter()
    violations = Counter()
    examples = defaultdict(list)
    details = []
    digest = hashlib.sha256()
    with ZipFile(args.archive) as archive:
        pred_name = next(n for n in archive.namelist() if n.endswith('/predictions.jsonl'))
        root = pred_name.rsplit('/', 1)[0]
        reported = json.loads(archive.read(root + '/reports/scored/summary.json'))
        meta = json.loads(archive.read(root + '/run_metadata.json'))
        seen = set()
        for line in archive.open(pred_name):
            digest.update(line)
            r = json.loads(line)
            rid = r['id']
            assert rid in gold and rid not in seen
            seen.add(rid)
            g = gold[rid]
            em, f1 = answer_scores(r['predicted_answer'], [g['answer'], *g['answer_aliases']])
            truth = {p['idx'] for p in g['paragraphs'] if p['is_supporting']}
            complete = truth <= set(r['predicted_support_idxs'])
            abstain = not r['predicted_answer'].strip() and not r.get('error')
            trace = r.get('iteration_trace') or []
            warning = bool(r.get('controller_format_warning_count'))
            counts['samples'] += 1
            counts['errors'] += bool(r.get('error'))
            counts['abstentions'] += abstain
            counts['explicit_insufficient_information'] += r.get('reader_raw_response', '').strip() == 'INSUFFICIENT_INFORMATION' if r.get('reader_raw_response') is not None else False
            counts['abstentions_with_complete_paragraphs'] += abstain and complete
            counts['abstentions_after_controller_answerable'] += abstain and r['stop_reason'] == 'answerable'
            counts['normalized_samples'] += warning
            counts['reader_output_limit_hits'] += r.get('reader_output_tokens') == meta['identity']['reader']['max_new_tokens']
            counts['embedding_truncations'] += len(r.get('embedding', {}).get('truncations', []))
            reasons[r['stop_reason']] += 1
            refs = [normalize(a) for a in [g['answer'], *g['answer_aliases']] if normalize(a)]
            raw_has = any(a in normalize(r['raw_context']) for a in refs)
            compressed_has = any(a in normalize(r['compressed_context']) for a in refs)
            counts['raw_answer_substring_present'] += raw_has
            counts['compressed_answer_substring_present'] += compressed_has
            counts['abstention_compressed_answer_substring_present'] += abstain and compressed_has
            previous = set()
            queries = set()
            catalog = {s['id']: s for s in r.get('segment_catalog', [])}
            for index, t in enumerate(trace):
                counts['iterations'] += 1
                qkey = ' '.join(t['query'].casefold().split())
                counts['exact_repeated_query_iterations'] += qkey in queries
                queries.add(qkey)
                selected = set(t['accumulated_evidence_ids'])
                counts['no_new_iterations'] += not bool(selected - previous)
                violations['nonmonotonic_union'] += bool(previous - selected)
                violations['union_mismatch'] += selected != previous | set(t['retrieved_evidence_ids'])
                violations['initial_query_mismatch'] += index == 0 and t['query'] != g['question']
                violations['followup_query_mismatch'] += index > 0 and t['query'] != trace[index-1]['follow_up_question']
                rankings = t.get('ranked_evidence') or []
                scores = sorted(s['dual'] for s in rankings)
                if scores:
                    percentile_index = (len(scores)-1)*.9
                    low, high = math.floor(percentile_index), math.ceil(percentile_index)
                    threshold = scores[low] + (percentile_index-low)*(scores[high]-scores[low])
                    violations['percentile_threshold'] += abs(threshold-t['threshold']) > 1e-10
                violations['incomplete_ranking'] += len(rankings) != len(catalog) or {s['id'] for s in rankings} != set(catalog)
                violations['score_formula'] += any(abs(s['dual'] - (.6*s['semantic']+.4*s['lexical'])) > 1e-10 for s in rankings)
                violations['threshold_selection'] += {s['id'] for s in rankings if s['dual'] >= t['threshold']} != set(t['retrieved_evidence_ids'])
                if index == 0:
                    first_p = {catalog[s]['paragraph_idx'] for s in selected}
                    counts['first_iteration_complete_paragraphs'] += truth <= first_p
                previous = selected
            if trace:
                violations['final_selection'] += previous != set(r['selected_evidence_ids'])
                violations['final_provenance'] += {catalog[s]['paragraph_idx'] for s in previous} != set(r['predicted_support_idxs'])
                counts['max_iterations_with_complete_paragraphs'] += r['stop_reason']=='max_iterations' and complete
                counts['answerable_missing_gold_paragraphs'] += r['stop_reason']=='answerable' and not complete
            row = dict(id=rid, hop=len(g['question_decomposition']), em=em, f1=f1,
                       complete_paragraphs=complete, abstain=abstain, error=bool(r.get('error')),
                       stop_reason=r['stop_reason'], format_warning=warning,
                       raw_answer_substring_present=raw_has, compressed_answer_substring_present=compressed_has,
                       iterations=r['iterations'], compression_ratio=r['compressed_context_tokens']/r['raw_context_tokens'],
                       compressed_context_tokens=r['compressed_context_tokens'],
                       compression_seconds=r['compression_seconds'], reader_seconds=r['reader_seconds'])
            details.append(row)
            for group in ('all', 'hop_'+str(row['hop']), 'stop_'+r['stop_reason'],
                          'coverage_'+str(complete), 'abstain_'+str(abstain), 'normalized_'+str(warning)):
                groups[group].append(row)
            for label, condition in [('abstain_with_answer_substring',abstain and compressed_has),
                                     ('wrong_with_complete_paragraphs',not em and complete and not abstain),
                                     ('normalized_controller',warning), ('error',bool(r.get('error')))]:
                if condition and len(examples[label])<8:
                    examples[label].append({'id':rid, 'question':g['question'], 'gold_answer':g['answer'],
                                            'predicted_answer':r['predicted_answer'], 'stop_reason':r['stop_reason'],
                                            'error':r.get('error'), 'context':r['compressed_context'],
                                            'controller_calls':[{k:c.get(k) for k in ('raw_response','format_warnings','parse_error')} for c in r.get('controller_calls',[])]})
        assert seen == set(gold)
        assert digest.hexdigest() == meta['output_sha256']
    def summary(rows):
        return {'n':len(rows), **{k:statistics.fmean(r[k] for r in rows) for k in
            ('em','f1','complete_paragraphs','abstain','iterations','compression_ratio','compressed_context_tokens','compression_seconds','reader_seconds')}}
    result = {'archive_sha256':sha256(args.archive), 'dataset_sha256':meta['identity']['dataset_sha256'],
              'predictions_sha256_verified':digest.hexdigest(), 'gold_sha256':sha256(args.gold),
              'reported_overall':reported['overall'], 'counts':dict(counts), 'stop_reasons':dict(reasons),
              'trace_violations':dict(violations), 'groups':{k:summary(v) for k,v in groups.items()},
              'examples':dict(examples),
              'notes':['Substring presence is a diagnostic heuristic, not entailment.',
                       'Paragraph coverage is not sentence sufficiency.',
                       'Group comparisons are observational, not causal effects.',
                       'Trace checks verify saved arithmetic and bookkeeping, not neural representation correctness or exact paper equivalence.',
                       'No inference performed; predictions and gold used only for offline diagnosis.']}
    assert abs(result['groups']['all']['em']-reported['overall']['answer_em'])<1e-12
    assert abs(result['groups']['all']['f1']-reported['overall']['answer_f1'])<1e-12
    out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'per_sample.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    print(json.dumps({k:result[k] for k in ('counts','stop_reasons','trace_violations','groups')},indent=2))


if __name__ == '__main__':
    main()
