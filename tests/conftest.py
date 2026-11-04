import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def sc_root():
    # Set SC_ROOT to the extracted Speech Commands v2 folder to run data tests
    root = os.environ.get("SC_ROOT")
    if not root or not Path(root, "testing_list.txt").exists():
        pytest.skip("SC_ROOT not set to an extracted Speech Commands v2 folder")
    return Path(root)


def hf_or_skip():
    # Parity tests need transformers + network access to the Hugging Face Hub (or a warm cache)
    pytest.importorskip("transformers")
    try:
        from ast_reference import load_ast_into_tat
        return load_ast_into_tat()
    except OSError as e:
        pytest.skip(f"cannot load AST from the Hub: {e}")
