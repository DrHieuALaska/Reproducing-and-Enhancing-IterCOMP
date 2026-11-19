"""Cloud setup regressions using fake Hub/secret clients, without model imports."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from itercomp.cloud import configure_hf_auth

ROOT = Path(__file__).resolve().parents[1]


class CloudTests(unittest.TestCase):
    def hub(self, token=None):
        api = Mock()
        return SimpleNamespace(get_token=Mock(return_value=token), HfApi=Mock(return_value=api)), api

    def test_existing_login_is_exported_without_printing_token(self):
        hub, api = self.hub("synthetic-private-token")
        out = io.StringIO()
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.dict(os.environ, {}, clear=True), redirect_stdout(out):
            result = configure_hf_auth("kaggle")
            self.assertEqual(os.environ["HF_TOKEN"], "synthetic-private-token")
        self.assertTrue(result["authenticated"])
        api.whoami.assert_called_once()
        self.assertNotIn("synthetic-private-token", out.getvalue())
        self.assertNotIn("synthetic-private-token", json.dumps(result))

    def test_kaggle_secret(self):
        hub, api = self.hub()
        secrets = SimpleNamespace(UserSecretsClient=Mock(return_value=SimpleNamespace(
            get_secret=Mock(return_value="  synthetic-secret  "))))
        with patch.dict(sys.modules, {"huggingface_hub": hub, "kaggle_secrets": secrets}), patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            result = configure_hf_auth("kaggle")
            self.assertEqual(os.environ["HF_TOKEN"], "synthetic-secret")
        self.assertEqual(result["source"], "kaggle HF_TOKEN secret")
        api.whoami.assert_called_once()

    def test_missing_secret_guidance_does_not_leak_exception(self):
        hub, _ = self.hub()
        secrets = SimpleNamespace(UserSecretsClient=Mock(side_effect=RuntimeError("sensitive exception detail")))
        out = io.StringIO()
        with patch.dict(sys.modules, {"huggingface_hub": hub, "kaggle_secrets": secrets}), redirect_stdout(out):
            result = configure_hf_auth("kaggle")
        self.assertFalse(result["authenticated"])
        self.assertEqual(result["secret_error"], "RuntimeError")
        self.assertIn("Add-ons > Secrets", out.getvalue())
        self.assertNotIn("sensitive exception detail", out.getvalue())

    def test_offline_secret_does_not_make_network_calls(self):
        hub, api = self.hub()
        userdata = SimpleNamespace(get=Mock(return_value="synthetic-colab-token"))
        with patch.dict(sys.modules, {"huggingface_hub": hub, "google.colab": SimpleNamespace(userdata=userdata)}), patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            result = configure_hf_auth("colab", online=False)
        self.assertTrue(result["token_available"])
        self.assertFalse(result["authenticated"])
        api.whoami.assert_not_called()

    def test_invalid_token_is_not_silently_accepted_or_exposed(self):
        hub, api = self.hub("synthetic-invalid-token")
        api.whoami.side_effect = RuntimeError("sensitive provider response")
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "authentication could not be validated") as raised:
                configure_hf_auth("kaggle")
        self.assertNotIn("synthetic-invalid-token", str(raised.exception))
        self.assertNotIn("sensitive provider response", str(raised.exception))

    def test_torch_only_configuration_precedes_optional_imports(self):
        script = """import os, sys
os.environ.update(USE_TF='1', USE_TORCH='0', USE_FLAX='1')
sys.path.insert(0, 'src')
import itercomp
assert os.environ['USE_TF'] == '0'
assert os.environ['USE_TORCH'] == '1'
assert os.environ['USE_FLAX'] == '0'
assert not {'tensorflow', 'flax', 'torch', 'transformers', 'FlagEmbedding'} & set(sys.modules)
"""
        result = subprocess.run([sys.executable, "-S", "-c", script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
