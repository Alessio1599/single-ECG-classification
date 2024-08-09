# tests/test_utils.py

import numpy as np
import pytest # pytest is designed to automatically discover and run tests based on certain conventions and configurations.
from utils.utils import load_data
from models.CNN import build_CNN_for

def test_load_data_shapes():
    """Test that load_data returns data with the expected shapes."""
    x_train, y_train, x_test, y_test = load_data()
    assert x_train.shape == (87554, 187), f"Unexpected x_train shape: {x_train.shape}"
    assert y_train.shape[0] == 87554, f"Unexpected y_train shape: {y_train.shape}"
    assert x_test.shape == (21892, 187), f"Unexpected x_test shape: {x_test.shape}"
    assert y_test.shape[0] == 21892, f"Unexpected y_test shape: {y_test.shape}"

