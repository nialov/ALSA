"""
pytest fixtures for use in tests.
"""
from pathlib import Path

import pytest

from alsa import crack_train


@pytest.fixture
def setup_train_test(tmp_path: Path):
    """
    Setup training data for test(s).
    """
    assert len(list(tmp_path.iterdir())) == 0

    # Setup directories
    crack_train.train_directory_setup(tmp_path)

    return tmp_path
