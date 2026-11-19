"""Original IterCOMP algorithm. Importing this package never loads a model."""

import os

# This implementation uses only PyTorch. Set these before any lazy Transformers
# or FlagEmbedding import, including in standalone experiment subprocesses.
# Otherwise unrelated, preinstalled TensorFlow/Flax packages can break imports.
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

__version__ = "0.1.0"
