"""Optional research diagnostics for MuSiQue-Ans / IterCOMP. Standard library only."""
import argparse
import csv
import math
import random
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from benchmark import read_jsonl, write_jsonl, dump, support, hops, validate, answer_scores, avg, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index(path):
    rows = {}
    for r in read_jsonl(path):
        require(r['id'] not in rows, f"Duplicate ID: {r['id']}")
        rows[r['id']] = r
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)


def metric_summary(rows):
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k not in ('hop', 'iteration')})
    return {'n': len(rows), 'means': {k: avg([r[k] for r in rows if r.get(k) is not None]) for k in keys},
            'counts': {k: sum(r.get(k) is not None for r in rows) for k in keys}}


def grouped(rows, field):
    gr = defaultdict(list)
    for r in rows: gr[str(r[field])].append(r)
    return {k: metric_summary(v) for k, v in sorted(gr.items())}


def paragraph_metrics(pred, gold, prefix='support'):
    tp = len(pred & gold)
    return {f'{prefix}_precision': tp / len(pred) if pred else 0.0,
            f'{prefix}_recall': tp / len(gold),
            f'{prefix}_f1': 2 * tp / (len(pred) + len(gold)),
            f'{prefix}_complete': int(gold <= pred)}


def structure(gold):
    """Heuristic topology from #1, #2 ... placeholders; never feed into the model."""
    qs = gold['question_decomposition']; n = len(qs)
    edges = set()
    for i, q in enumerate(qs, 1):
        for ref in re.findall(r'#(\d+)\b', q['question']):
            j = int(ref)
            if not 1 <= j < i: return 'unknown'
            edges.add((j, i))
    if edges == {(i, i+1) for i in range(1, n)}: return 'chain'
    # No explicit placeholders may mean a comparison or missing formatting.
    if not edges: return 'no_explicit_dependencies'
    return 'nonlinear_or_partial'


def query_key(q):
    return ' '.join(re.findall(r'\w+', q.casefold()))


def idset(values, allowed, label):
    require(isinstance(values, list) and all(isinstance(v, str) for v in values), f'{label}: expected list of strings')
    require(len(values) == len(set(values)), f'{label}: duplicate IDs')
    result = set(values)
    require(result <= allowed, f'{label}: unknown IDs {result-allowed}')
    return result


def numeric(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value), f'{label}: expected finite number')
    return value


def diagnose(gold_path, pred_path, out, annotations=None):
    golds, preds = index(gold_path), index(pred_path)
    require(set(golds)==set(preds) and golds, 'Prediction and gold IDs must match exactly')
    labels = {}
    for r in read_jsonl(annotations) if annotations else []:
        key = (r['id'], r['iteration']); require(key not in labels, f'Duplicate annotation {key}')
        for field in ('sufficient', 'follow_up_useful'):
            require(r.get(field) is None or isinstance(r[field], bool), f'{key}: invalid {field}')
        labels[key] = r
    used_labels = set(); samples=[]; iters=[]; untraced=[]; annotation_queue=[]
    for rid, g in golds.items():
        validate(g); p=preds[rid]; truth=support(g); valid_p={x['idx'] for x in g['paragraphs']}
        em, f1=answer_scores(p['predicted_answer'], [g['answer'],*g['answer_aliases']])
        pred=set(p['predicted_support_idxs']); require(pred<=valid_p, f'{rid}: invalid paragraph IDs')
        base={'id':rid, 'hop':hops(g), 'structure':structure(g), 'answer_em':em, 'answer_f1':f1,
              **paragraph_metrics(pred,truth)}
        # This is a diagnostic bucket, NOT a proven cause of an incorrect answer.
        base['failure_bucket'] = ('exact_match' if em else 'missing_gold_paragraph' if not truth<=pred else 'all_gold_paragraphs_present')
        if p.get('context_token_budget') is not None:
            budget=numeric(p['context_token_budget'], f'{rid}: budget');require(budget>=0,f'{rid}: negative budget')
            if p.get('compressed_context_tokens') is not None:
                count=numeric(p['compressed_context_tokens'],f'{rid}: compressed_context_tokens')
                base['budget_violation']=int(count>budget);base['budget_overrun_tokens']=max(0,count-budget)
        trace=p.get('iteration_trace')
        if trace is None:
            untraced.append(rid); samples.append(base); continue
        require(isinstance(trace,list) and trace, f'{rid}: trace must be nonempty if supplied')
        catalog={}
        for e in p['segment_catalog']:
            sid=e['id']; require(isinstance(sid,str) and sid not in catalog, f'{rid}: invalid/duplicate segment ID')
            require(e['paragraph_idx'] in valid_p, f'{rid}: unknown source paragraph')
            catalog[sid]=e
        allowed=set(catalog); previous=set(); seen_queries=set(); local=[]
        for h,t in enumerate(trace,1):
            require(t['iteration']==h, f'{rid}: nonconsecutive iteration index')
            require(isinstance(t['answerable'],bool), f'{rid}/{h}: invalid controller judgment')
            require(isinstance(t['query'],str) and t['query'].strip(), f'{rid}/{h}: invalid query')
            qkey=query_key(t['query']); follow=t.get('follow_up_question')
            require(follow is None or isinstance(follow,str), f'{rid}/{h}: invalid follow-up')
            selected=idset(t['accumulated_evidence_ids'],allowed,f'{rid}/{h}: accumulated')
            retrieved=idset(t['retrieved_evidence_ids'],allowed,f'{rid}/{h}: retrieved')
            para={catalog[s]['paragraph_idx'] for s in selected}; retpara={catalog[s]['paragraph_idx'] for s in retrieved}
            new=selected-previous; removed=previous-selected; oldpara={catalog[s]['paragraph_idx'] for s in previous}
            r={'id':rid,'hop':hops(g),'structure':base['structure'],'iteration':h,
               'query':t['query'],'follow_up_question':follow,'controller_answerable':int(t['answerable']),
               'repeated_query':int(qkey in seen_queries),'retrieved_segments':len(retrieved),'retained_segments':len(selected),
               'new_segments':len(new),'removed_segments':len(removed),'no_new_segments':int(not new),
               'new_gold_paragraphs':len((para-oldpara)&truth),'retained_paragraphs':len(para),
               **paragraph_metrics(para,truth),**paragraph_metrics(retpara,truth,'retrieval'),
               'stop_with_missing_gold_paragraph':int(t['answerable'] and not truth<=para)}
            for key in ('context_tokens','controller_input_tokens','controller_output_tokens','controller_seconds','filter_seconds'):
                if t.get(key) is not None:
                    value=numeric(t[key],f'{rid}/{h}: {key}'); require(value>=0,f'{rid}/{h}: negative {key}');r[key]=value
            ranking=t.get('ranked_evidence')
            if ranking is not None:
                ranked_ids=[item['id'] for item in ranking]; idset(ranked_ids,allowed,f'{rid}/{h}: ranking')
                for item in ranking:
                    for k in ('semantic','lexical','dual'): numeric(item[k],f'{rid}/{h}: {k}')
                require(all(a['dual']>=b['dual'] for a,b in zip(ranking,ranking[1:])), f'{rid}/{h}: ranking must be descending by dual score')
                for k in (5,10,20):
                    if len(ranking)>=k or t.get('ranking_complete') is True:
                        top={catalog[s]['paragraph_idx'] for s in ranked_ids[:k]}
                        r[f'paragraph_recall_at_{k}']=len(top&truth)/len(truth)
                if t.get('ranking_complete') is True:
                    require(set(ranked_ids)==allowed, f'{rid}/{h}: incomplete candidate ranking')
                    for k in ('semantic','lexical','dual'):
                        vs=[x[k] for x in ranking]
                        if vs:
                            r[f'{k}_mean']=avg(vs);r[f'{k}_std']=statistics.pstdev(vs);r[f'{k}_range']=max(vs)-min(vs)
                    if ranking: r['ranked_candidates']=len(ranking)
            key=(rid,h); label=labels.get(key,{})
            annotation_queue.append({'id':rid,'iteration':h,'question':g['question'],'information':t.get('controller_context'), 'selected_segments':[catalog[s] for s in t['accumulated_evidence_ids']], 'sufficient':None})
            if label:
                used_labels.add(key)
                if label.get('sufficient') is not None:
                    y=label['sufficient']; pred_y=t['answerable']; r['annotated_sufficient']=int(y)
                    r['judgment_correct']=int(y==pred_y)
                    r['judgment_false_positive']=int(pred_y) if not y else None
                    r['judgment_false_negative']=int(not pred_y) if y else None
                if label.get('follow_up_useful') is not None:
                    r['annotated_follow_up_useful']=int(label['follow_up_useful'])
            seen_queries.add(qkey);previous=selected;local.append(r)
        final=idset(p['selected_evidence_ids'],allowed,f'{rid}: final selection')
        require(final==previous,f'{rid}: final selected evidence differs from last accumulated state')
        require({catalog[s]['paragraph_idx'] for s in final}==pred,f'{rid}: paragraph prediction differs from final source IDs')
        if p.get('iterations') is not None: require(p['iterations']==len(trace),f'{rid}: iteration count mismatch')
        reason=p.get('stop_reason','unspecified');base.update({'stop_reason':reason,'trace_iterations':len(trace),
            'repeated_query_rate':avg([x['repeated_query'] for x in local]),
            'no_new_segment_rate':avg([x['no_new_segments'] for x in local]),
            'final_controller_answerable':local[-1]['controller_answerable'],
            'final_judgment_missing_gold_proxy':local[-1]['stop_with_missing_gold_paragraph'],
            'total_removed_segments':sum(x['removed_segments'] for x in local)})
        if all(x.get('context_tokens') is not None for x in local): base['peak_retained_context_tokens']=max(x['context_tokens'] for x in local)
        samples.append(base);iters.extend(local)
    require(set(labels)==used_labels,'Some annotations do not match traced states')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'diagnostic_per_sample.csv',samples);write_csv(out/'per_iteration.csv',iters)
    write_jsonl(out/'annotation_queue.jsonl',annotation_queue)
    dump(out/'diagnostic_summary.json',{'samples':metric_summary(samples),'by_hop':grouped(samples,'hop'),
        'by_structure':grouped(samples,'structure'),'iterations_by_step':grouped(iters,'iteration'),
        'iteration_metrics':metric_summary(iters),'trace_coverage':{'traced':len(samples)-len(untraced),'total':len(samples)},
        'failure_buckets':dict(Counter(r['failure_bucket'] for r in samples)),
        'stop_reasons':dict(Counter(r.get('stop_reason','no_trace') for r in samples)),
        'notes':['Paragraph coverage is not evidence sufficiency.', 'Judgment FPR/FNR use only explicit sufficiency annotations.', 'Late-iteration means are conditional on reaching those iterations.']})
    print(f'Diagnosed {len(samples)} samples, {len(iters)} iteration states; output: {out}')


def prepare_probes(gold_path, out, per_hop, seed):
    require(per_hop>0,'per-hop must be positive')
    rng=random.Random(seed); groups=defaultdict(list)
    for g in index(gold_path).values():
        validate(g);groups[hops(g)].append(g)
    inputs=[]; labels=[]
    def add(g, variant, paras, expected, status):
        rid=f'probe_{len(inputs):06d}'
        inputs.append({'id':rid,'question':g['question'],'paragraphs':[{k:p[k] for k in ('idx','title','paragraph_text')} for p in paras]})
        labels.append({'id':rid,'source_id':g['id'],'hop':hops(g),'variant':variant,'expected_answerable':expected,'label_status':status,
                       'answer':g['answer'],'answer_aliases':g['answer_aliases']})
    counts={}
    for h, rows in sorted(groups.items()):
        chosen=rng.sample(rows,min(per_hop,len(rows)));counts[h]=len(chosen)
        for g in chosen:
            truth=support(g); allp=g['paragraphs']; goldp=[p for p in allp if p['idx'] in truth]; victim=rng.choice(sorted(truth))
            add(g,'raw_full',allp,True,'dataset_assumption')
            add(g,'gold_only',goldp,True,'dataset_assumption')
            add(g,'raw_drop_one_support',[p for p in allp if p['idx']!=victim],False,'provisional')
            add(g,'gold_drop_one_support',[p for p in goldp if p['idx']!=victim],False,'provisional')
    # Small synthetic stress set: isolated sentence is ambiguous; both sentences resolve it.
    cases=[('What does my dog love?','My dog is Jenny.','It loves bones.','bones'),
           ('Where was the architect born?','The architect was Mira Chen.','She was born in Oslo.','Oslo'),
           ('What did the company acquire?','The company was called Northstar.','It acquired a bakery.','a bakery'),
           ('Who won the contest?','The contest featured two finalists, Imani and Noor.','The former won.','Imani'),
           ('What color was box A?','There were two boxes, A and B, in that order.','The former was blue.','blue'),
           ('Who received the grant?','The applicants were Kim and Lee.','The latter received the grant.','Lee')]
    # First/last positional referring expression tests should name the relevant antecedent in the question.
    for i,(q,a,b,answer) in enumerate(cases):
        for variant,context,expected in [('coreference_full',a+' '+b,True),('coreference_fragment',b,False)]:
            rid=f'probe_{len(inputs):06d}'
            inputs.append({'id':rid,'question':q,'paragraphs':[{'idx':0,'title':'','paragraph_text':context}]})
            labels.append({'id':rid,'source_id':f'synthetic_{i}','hop':None,'variant':variant,'expected_answerable':expected,'label_status':'synthetic_design','answer':answer,'answer_aliases':[]})
    # All designed synthetic cases require grounding, not merely guessing an answer string.
    order=list(range(len(inputs)));rng.shuffle(order)
    remapped_inputs=[];remapped_labels=[]
    for i,j in enumerate(order):
        rid=f'probe_{i:06d}';remapped_inputs.append({**inputs[j],'id':rid});remapped_labels.append({**labels[j],'id':rid})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/'inputs.jsonl',remapped_inputs);write_jsonl(out/'gold.jsonl',remapped_labels)
    dump(out/'manifest.json',{'source_sha256':sha256(gold_path),'seed':seed,'source_examples_by_hop':counts,'variants':dict(Counter(r['variant'] for r in labels)),'warning':'Drop-one labels are provisional: other paragraphs may contain alternative evidence. Audit manually before treating as true negatives.'})
    print(f'Created {len(inputs)} probes at {out}')


def score_probes(gold_path, pred_path, out, verified=None):
    gs,ps=index(gold_path),index(pred_path);require(gs and set(gs)==set(ps),'Prediction and probe IDs must match exactly')
    overrides=index(verified) if verified else {};require(set(overrides)<=set(gs),'Unknown verified probe IDs')
    rows=[]
    for rid,g in gs.items():
        lab=overrides.get(rid,{});y=lab.get('expected_answerable',g['expected_answerable']);p=ps[rid]['predicted_answerable']
        require(isinstance(y,bool) and isinstance(p,bool),f'{rid}: invalid judgment')
        status='human_verified' if rid in overrides else g['label_status']
        r={'id':rid,'source_id':g['source_id'],'variant':g['variant'],'label_status':status,'expected_answerable':int(y),
           'predicted_answerable':int(p),'correct':int(y==p),'false_positive':int(p) if not y else None,'false_negative':int(not p) if y else None}
        if y and 'predicted_answer' in ps[rid]:
            r['answer_em'],r['answer_f1']=answer_scores(ps[rid]['predicted_answer'],[g['answer'],*g['answer_aliases']])
        rows.append(r)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);write_csv(out/'probe_scores.csv',rows)
    # Never combine provisional and verified labels into an ostensibly reliable overall metric.
    # Matched pairs; keep label provenance of both states visible.
    by_source=defaultdict(dict)
    for r in rows: by_source[r['source_id']][r['variant']]=r
    pair_rows=[]
    for sid, variants in by_source.items():
        for a,b in [('raw_full','raw_drop_one_support'),('gold_only','gold_drop_one_support'),('coreference_full','coreference_fragment')]:
            if a not in variants or b not in variants:continue
            x,z=variants[a],variants[b]
            pair_rows.append({'source_id':sid,'pair':a+' vs '+b,'label_status':x['label_status']+' / '+z['label_status'],
                              'both_judgments_correct':int(x['correct'] and z['correct']),
                              'flips_to_unanswerable':int(x['predicted_answerable'] and not z['predicted_answerable'])})
    write_csv(out/'probe_pairs.csv',pair_rows)
    dump(out/'probe_summary.json',{'paired_by_variant_and_status':grouped([{**r,'group':r['pair']+' / '+r['label_status']} for r in pair_rows],'group'),'by_label_status':grouped(rows,'label_status'),'by_variant_and_status':grouped([{**r,'group':r['variant']+' / '+r['label_status']} for r in rows],'group'),
          'notes':['False positive rate denominator is insufficient states only; false negative rate denominator is sufficient states only.','Provisional labels are not ground truth.','Synthetic probes are a sanity check, not representative MuSiQue scores.']})
    print(f'Scored {len(rows)} probes at {out}')


def compare(left,right,out,seed=42,reps=2000):
    require(reps>0,'bootstrap repetitions must be positive')
    def load(p):
        rows={}
        with Path(p).open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                require(r['id'] not in rows,f'Duplicate ID: {r["id"]}');rows[r['id']]=r
        return rows
    a,b=load(left),load(right);require(a and set(a)==set(b),'Comparison requires identical sample IDs')
    rng=random.Random(seed);result={}
    for k in ('answer_em','answer_f1','support_precision','support_recall','support_f1','all_support_recovered','compression_ratio','total_llm_tokens','total_seconds'):
        pairs=[(float(a[i][k]),float(b[i][k])) for i in sorted(a) if a[i].get(k) not in (None,'') and b[i].get(k) not in (None,'')]
        if not pairs:continue
        diffs=[y-x for x,y in pairs]; n=len(diffs);boot=sorted(statistics.fmean(rng.choices(diffs,k=n)) for _ in range(reps))
        result[k]={'n_paired':n,'left_mean':avg([x for x,y in pairs]),'right_mean':avg([y for x,y in pairs]),'delta_right_minus_left':avg(diffs),
                   'bootstrap_95_ci':[boot[int(.025*(reps-1))],boot[int(.975*(reps-1))]]}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);dump(out,{'seed':seed,'repetitions':reps,'n_ids':len(a),'metrics':result,'notes':'Paired percentile bootstrap over questions. Interpret only compatible model/settings and complete telemetry.'})
    print(f'Comparison saved: {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    d=s.add_parser('diagnose');d.add_argument('--gold',required=True);d.add_argument('--predictions',required=True);d.add_argument('--out',required=True);d.add_argument('--annotations')
    d=s.add_parser('prepare-probes');d.add_argument('--gold',required=True);d.add_argument('--out',required=True);d.add_argument('--per-hop',type=int,default=20);d.add_argument('--seed',type=int,default=42)
    d=s.add_parser('score-probes');d.add_argument('--gold',required=True);d.add_argument('--predictions',required=True);d.add_argument('--out',required=True);d.add_argument('--verified')
    d=s.add_parser('compare');d.add_argument('--left',required=True);d.add_argument('--right',required=True);d.add_argument('--out',required=True);d.add_argument('--seed',type=int,default=42);d.add_argument('--reps',type=int,default=2000)
    a=p.parse_args()
    if a.cmd=='diagnose':diagnose(a.gold,a.predictions,a.out,a.annotations)
    elif a.cmd=='prepare-probes':prepare_probes(a.gold,a.out,a.per_hop,a.seed)
    elif a.cmd=='score-probes':score_probes(a.gold,a.predictions,a.out,a.verified)
    else:compare(a.left,a.right,a.out,a.seed,a.reps)

if __name__=='__main__':main()
