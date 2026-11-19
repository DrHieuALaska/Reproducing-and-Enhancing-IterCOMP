"""Mocked native-library checks: no GPU code or checkpoint loading."""
import importlib.metadata
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from itercomp.backends.quantization import check_quantization_runtime
from itercomp.config import LLMConfig


class QuantizationTests(unittest.TestCase):
    def fake_modules(self, native=True):
        return {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                         version=SimpleNamespace(cuda="13.0")),
                "bitsandbytes": SimpleNamespace(),
                "bitsandbytes.cextension": SimpleNamespace(lib=SimpleNamespace(compiled_with_cuda=native))}

    def test_old_cuda13_build_fails_before_native_import(self):
        modules = self.fake_modules()
        with patch.dict("sys.modules", modules), patch("itercomp.backends.quantization.importlib.metadata.version", return_value="0.45.5"):
            with self.assertRaisesRegex(RuntimeError, "older than the selected CUDA-13-compatible pin"):
                check_quantization_runtime(LLMConfig())

    def test_cpu_only_or_missing_native_library_rejected(self):
        for lib in (None, SimpleNamespace(compiled_with_cuda=False)):
            modules = self.fake_modules()
            modules["bitsandbytes.cextension"] = SimpleNamespace(lib=lib)
            with self.subTest(lib=lib), patch.dict("sys.modules", modules), patch("itercomp.backends.quantization.importlib.metadata.version", return_value="0.49.2"):
                with self.assertRaisesRegex(RuntimeError, "did not load its native CUDA backend"):
                    check_quantization_runtime(LLMConfig())

    def test_matching_native_backend_accepted(self):
        with patch.dict("sys.modules", self.fake_modules()), patch("itercomp.backends.quantization.importlib.metadata.version", return_value="0.49.2"):
            result = check_quantization_runtime(LLMConfig())
        self.assertEqual(result, {"bitsandbytes_version": "0.49.2", "torch_cuda": "13.0", "native_cuda_backend": True})

    def test_no_quantization_and_mocks_require_no_backend_imports(self):
        with patch("itercomp.backends.quantization.importlib.metadata.version", side_effect=AssertionError("unexpected lookup")):
            self.assertIsNone(check_quantization_runtime(LLMConfig(quantization="none")))
            self.assertIsNone(check_quantization_runtime(LLMConfig(backend="mock")))

    def test_missing_package_has_installation_guidance(self):
        with patch.dict("sys.modules", self.fake_modules()), patch("itercomp.backends.quantization.importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "bitsandbytes is not installed"):
                check_quantization_runtime(LLMConfig())


if __name__ == "__main__":
    unittest.main()
