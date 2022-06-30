from importlib.resources import path
import numpy as np
import pytest
import sys

import torch

sys.path.append(sys.path[0] + '\\..')

from model_functions.loss_function import numpy_single_custom_loss_function, torch_single_custom_loss_function


@pytest.mark.parametrize("test_input, test_label, expected", [(0.0,0.5,0.5), (0.1,0.2,0.1), (0.1,0.55,0.45), (0.1,0.85,0.25), 
(0.1,0.9,0.2), (0.4,0.89,0.49),(0.4,0.91,0.49), (0.75,0.85,0.1), (0.9,0.99,0.09), (0.903,0.270,0.367), (-1.0,2.0,2.0)])
def test_torch_single_custom_loss_function(test_input, test_label, expected):
    assert torch_single_custom_loss_function(torch.tensor([test_input]), torch.tensor([test_label])) == pytest.approx(expected, 0.000001)
    assert torch_single_custom_loss_function(torch.tensor([test_label]), torch.tensor([test_input])) == pytest.approx(expected, 0.000001)


@pytest.mark.parametrize("test_input, test_label, expected", [(0.0,0.5,0.5), (0.1,0.2,0.1), (0.1,0.55,0.45), (0.1,0.85,0.25), 
(0.1,0.9,0.2), (0.4,0.89,0.49),(0.4,0.91,0.49), (0.75,0.85,0.1), (0.9,0.99,0.09), (0.903,0.270,0.367), (-1.0,2.0,2.0)])
def test_numpy_single_custom_loss_function(test_input, test_label, expected):
    assert numpy_single_custom_loss_function(test_input, test_label) == pytest.approx(expected, 0.000001)
    assert numpy_single_custom_loss_function(test_label, test_input) == pytest.approx(expected, 0.000001)

def test_compare_numpy_and_torch_loss_function():
    for test_input in np.arange(0.0, 1.0, 0.01):
        for test_label in np.arange(0.0, 1.0, 0.01):
            assert numpy_single_custom_loss_function(test_input, test_label) == torch_single_custom_loss_function(torch.tensor([test_input]), torch.tensor([test_label]))

