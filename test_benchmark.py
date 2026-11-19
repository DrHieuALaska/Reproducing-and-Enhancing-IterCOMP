import csv
import json
import tempfile
import unittest
from pathlib import Path
from benchmark import write_jsonl, evaluate, read_jsonl
from diagnostics import diagnose, prepare_probes, score_probes, compare


def example():
    return {'id':'synthetic-1','question':'Where was the author born?','answer':'Paris','answer_aliases':[],'answerable':True,
            'paragraphs':[{'idx':i,'title':str(i),'paragraph_text':f'Sentence {i}.','is_supporting':i in (3,12)} for i in range(20)],
            'question_decomposition':[{'id':'a','question':'Who is the author?','answer':'Alice','paragraph_support_idx':3},{'id':'b','question':'Where was #1 born?','answer':'Paris','paragraph_support_idx':12}]}


def prediction():
    return {'id':'synthetic-1','predicted_answer':'Paris','predicted_support_idxs':[3,7,12],'predicted_answerable':True,
            'raw_context_tokens':100,'compressed_context_tokens':25,'iterations':2,'stop_reason':'answerable',
            'segment_catalog':[{'id':f'p{i}_s0','paragraph_idx':i,'sentence_idx':0,'text':f'Sentence {i}.'} for i in (3,7,12)],
            'selected_evidence_ids':['p3_s0','p7_s0','p12_s0'],
            'iteration_trace':[{'iteration':1,'query':'Where was the author born?','retrieved_evidence_ids':['p3_s0','p7_s0'],'accumulated_evidence_ids':['p3_s0','p7_s0'],'answerable':False,'follow_up_question':'Where was Alice born?'},
                               {'iteration':2,'query':'Where was Alice born?','retrieved_evidence_ids':['p12_s0'],'accumulated_evidence_ids':['p3_s0','p7_s0','p12_s0'],'answerable':True,'follow_up_question':None,'ranking_complete':True,'ranked_evidence':[{'id':f'p{i}_s0','semantic':s,'lexical':s,'dual':s} for i,s in [(12,0.9),(3,0.4),(7,0.1)]]}]}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.g=self.root/'gold.jsonl';self.p=self.root/'pred.jsonl';write_jsonl(self.g,[example()]);write_jsonl(self.p,[prediction()])
    def tearDown(self):self.tmp.cleanup()
    def test_main_and_diagnostics(self):
        out=self.root/'out';evaluate(self.g,self.p,out)
        s=json.loads((out/'summary.json').read_text())['overall'];self.assertEqual(s['answer_em'],1);self.assertAlmostEqual(s['support_precision'],2/3);self.assertEqual(s['weighted_compression_ratio'],.25);self.assertEqual(s['support_em'],0)
        ann=self.root/'annotations.jsonl';write_jsonl(ann,[{'id':'synthetic-1','iteration':1,'sufficient':False},{'id':'synthetic-1','iteration':2,'sufficient':True}]);diagnose(self.g,self.p,out,ann)
        with (out/'per_iteration.csv').open() as f:r=list(csv.DictReader(f))
        self.assertEqual(float(r[0]['support_recall']),.5);self.assertEqual(float(r[1]['support_recall']),1);self.assertEqual(float(r[1]['paragraph_recall_at_5']),1);self.assertEqual(float(r[0]['judgment_false_positive']),0);self.assertEqual(float(r[1]['judgment_false_negative']),0)
        compare(out/'per_sample.csv',out/'per_sample.csv',out/'comparison.json',reps=10);self.assertEqual(json.loads((out/'comparison.json').read_text())['metrics']['answer_f1']['delta_right_minus_left'],0)
    def test_bad_provenance(self):
        p=prediction();p['predicted_support_idxs']=[3];write_jsonl(self.p,[p])
        with self.assertRaises(ValueError):diagnose(self.g,self.p,self.root/'bad')
    def test_probes(self):
        out=self.root/'probes';prepare_probes(self.g,out,1,42);gs=list(read_jsonl(out/'gold.jsonl'));self.assertEqual(len(gs),16)
        self.assertTrue(all('expected_answerable' not in r and 'variant' not in r for r in read_jsonl(out/'inputs.jsonl')))
        write_jsonl(self.p,[{'id':g['id'],'predicted_answerable':g['expected_answerable']} for g in gs]);score_probes(out/'gold.jsonl',self.p,out)
        s=json.loads((out/'probe_summary.json').read_text());self.assertEqual(s['by_label_status']['provisional']['means']['false_positive'],0)
    def test_legacy_prediction(self):
        p=prediction();p.pop('iteration_trace');p.pop('segment_catalog');write_jsonl(self.p,[p]);diagnose(self.g,self.p,self.root/'legacy');self.assertEqual(json.loads((self.root/'legacy'/'diagnostic_summary.json').read_text())['trace_coverage']['traced'],0)

if __name__=='__main__':unittest.main()
