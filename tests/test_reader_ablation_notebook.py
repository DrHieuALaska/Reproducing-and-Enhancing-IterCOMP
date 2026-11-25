"""Execute notebook orchestration with synthetic responses, never model weights."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from benchmark import write_jsonl
from itercomp.backends.mock import MockBackend
from itercomp.experiment import run_compression, run_reader
from itercomp.serialization import file_hash
from tests.test_core import config, ROOT


NOTEBOOK = ROOT / 'notebooks/run_reader_calibration_6_conditions_kaggle.ipynb'


class SyntheticReader(MockBackend):
    calls = 0

    def __init__(self):
        super().__init__([])
        self.initialization['resolved_revision'] = 'synthetic-reader-revision'

    def generate(self, messages):
        type(self).calls += 1
        instruction = 'If the information is insufficient, return exactly INSUFFICIENT_INFORMATION.'
        self.responses = ['INSUFFICIENT_INFORMATION' if instruction in messages[0]['content'] else 'Paris']
        return super().generate(messages)


class ReaderAblationNotebookTests(unittest.TestCase):
    def test_discover_kaggle_extracted_export_and_explicit_directory(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cells = {c['id']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
        old_cwd = Path.cwd()
        with (tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()),
              ExitStack() as cleanup):
            cleanup.callback(os.chdir, old_cwd)
            root = Path(temp)
            attached = root/'input/datasets/account/arbitrary-dataset-name'
            compression = attached/'calibration_compression'
            compression.mkdir(parents=True)
            data = root/'data/calibration'
            data.mkdir(parents=True)
            (data/'inputs.jsonl').write_text('{}\n', encoding='utf-8')
            (data/'gold.jsonl').write_text('{}\n', encoding='utf-8')
            metadata = {'identity': {'stage': 'compress', 'method': 'itercomp',
                                    'dataset_sha256': file_hash(data/'inputs.jsonl')}}
            (compression/'run_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            (compression/'compression_results.jsonl').write_text('{}\n', encoding='utf-8')
            # Smoke metadata must not create another candidate.
            smoke = attached/'smoke_compression'
            smoke.mkdir()
            (smoke/'run_metadata.json').write_text(json.dumps({'identity': {
                'stage': 'compress', 'method': 'itercomp', 'dataset_sha256': 'different'}}), encoding='utf-8')
            (smoke/'compression_results.jsonl').write_text('{}\n', encoding='utf-8')
            original_files = {p.relative_to(attached): p.read_bytes() for p in attached.rglob('*') if p.is_file()}
            env = {}
            exec(cells['settings'], env)
            env.update(SOURCE_REPO=ROOT, INPUT_ROOT=root/'input', DATA_DIR=data.parent,
                       REPO=ROOT, OUTPUT_ROOT=root/'output')
            exec(cells['paths'], env)
            self.assertEqual(env['EXTRACTED'], attached.resolve())
            self.assertIsNone(env['archive_sha'])
            self.assertIsNotNone(env['directory_sha'])
            self.assertEqual(env['source_info']['format'], 'extracted_directory')
            self.assertEqual(original_files, {p.relative_to(attached): p.read_bytes()
                                             for p in attached.rglob('*') if p.is_file()})
            # Explicit directory also works when discovery has no candidates.
            env.update(CALIBRATION_DIR=attached, INPUT_ROOT=root/'not-mounted')
            exec(cells['paths'], env)
            self.assertEqual(env['EXTRACTED'], attached.resolve())

    def test_all_cells_compile_and_outputs_are_clear(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        ids = [c['id'] for c in notebook['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])

    def test_six_conditions_scoring_resume_and_export_with_synthetic_reader(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cells = {c['id']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
        old_cwd = Path.cwd()
        old_path = list(sys.path)
        SyntheticReader.calls = 0
        try:
            with (tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()),
                  ExitStack() as cleanup):
                cleanup.callback(os.chdir, old_cwd)
                root = Path(temp)
                data = root/'data/benchmark/calibration'
                data.mkdir(parents=True)
                template = json.loads((ROOT/'tests/fixtures/inputs.jsonl').read_text(encoding='utf-8'))
                truth = json.loads((ROOT/'tests/fixtures/gold.jsonl').read_text(encoding='utf-8'))
                inputs, gold = [], []
                for i in range(150):
                    row, label = deepcopy(template), deepcopy(truth)
                    row['id'] = label['id'] = f'synthetic-{i:03d}'
                    inputs.append(row)
                    gold.append(label)
                write_jsonl(data/'inputs.jsonl', inputs)
                write_jsonl(data/'gold.jsonl', gold)
                cfg = config()
                # The notebook expects a real reader config in the archive. The
                # factory is replaced below, so no real backend is constructed.
                cfg.reader.backend = 'transformers'
                cfg.reader.model_id = 'synthetic-reader'
                compressed = run_compression(cfg, data/'inputs.jsonl', root/'original/compression')
                with patch('itercomp.experiment.create_backend', return_value=SyntheticReader()):
                    run_reader(cfg, data/'inputs.jsonl', compressed, root/'original/reader')
                archive_path = root/'itercomp_runs_export_calibration.zip'
                with ZipFile(archive_path, 'w') as archive:
                    for path in (root/'original').rglob('*'):
                        if path.is_file():
                            archive.write(path, path.relative_to(root/'original').as_posix())

                env = {}
                def execute(name):
                    exec(compile(cells[name], name, 'exec'), env)

                execute('settings')
                env.update(SOURCE_REPO=ROOT, CALIBRATION_ZIP=archive_path,
                           DATA_DIR=data.parent, REPO=root/'working_repo',
                           OUTPUT_ROOT=root/'outputs', INSTALL_DEPENDENCIES=False)
                execute('paths')
                execute('dependencies')
                execute('artifacts')
                execute('conditions')
                original_hash = file_hash(env['COMPRESSION_RESULTS'])
                with (patch('itercomp.experiment.create_encoder', side_effect=AssertionError('encoder constructed')),
                      patch('itercomp.experiment.create_backend', side_effect=AssertionError('controller constructed'))):
                    execute('contexts')
                    execute('contexts')  # completed CPU baselines resume

                # Execute the real CLI in-process while replacing only model
                # construction. This exercises the notebook's subprocess driver,
                # metadata, six reader runs, and resume paths without inference.
                from scripts.run_experiment import main as run_main
                class FakeProcess:
                    def __init__(self, command, **kwargs):
                        captured = io.StringIO()
                        with (redirect_stdout(captured), patch('itercomp.experiment.create_encoder', side_effect=AssertionError('reader encoder')),
                              patch('itercomp.experiment.create_backend', side_effect=lambda _: SyntheticReader())):
                            run_main(command[2:])
                        self.stdout = io.StringIO(captured.getvalue())
                        self.returncode = 0

                    def wait(self):
                        return self.returncode

                    def poll(self):
                        return self.returncode

                    def terminate(self):
                        self.returncode = -1

                with patch('subprocess.Popen', FakeProcess):
                    before = SyntheticReader.calls
                    execute('readers')
                    self.assertEqual(SyntheticReader.calls-before, 900)
                    execute('readers')
                    self.assertEqual(SyntheticReader.calls-before, 900)
                real_run = subprocess.run
                with patch('subprocess.run', side_effect=lambda *a, **kw: real_run(
                        *a, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)):
                    execute('score')
                    execute('score')  # validated reports may be reused
                execute('comparison')
                execute('export')
                self.assertEqual(file_hash(env['COMPRESSION_RESULTS']), original_hash)
                report = json.loads((root/'outputs/comparison.json').read_text(encoding='utf-8'))
                self.assertEqual(len(report['conditions']), 6)
                for row in report['conditions']:
                    self.assertEqual(row['n'], 150)
                    self.assertEqual(row['errors'], 0)
                    self.assertEqual(row['EM'], 100 if 'no_abstention' in row['condition'] else 0)
                for row in report['paired_prompt_changes']:
                    self.assertEqual(row['exact_matches_gained'], 150)
                    self.assertEqual(row['abstentions_removed_net'], 150)
                self.assertEqual(report['archived_control_comparison']['raw_generation_changes'], 0)
                with ZipFile(root/'outputs_export.zip') as archive:
                    self.assertIn('comparison.csv', archive.namelist())
                    self.assertIn('per_sample_comparison.csv', archive.namelist())
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path


if __name__ == '__main__':
    unittest.main()
