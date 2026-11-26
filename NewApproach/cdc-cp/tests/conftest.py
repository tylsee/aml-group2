import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides=["loader.num_workers=0"])
