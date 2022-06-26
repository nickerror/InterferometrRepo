from importlib.resources import path
import pytest
import sys

sys.path.append(sys.path[0] + '\\..')

from model_functions.loss_function import abcdef, single_custom_loss_function
from model_functions.loss_function import custom_loss_function

def test_abcdef():
    assert abcdef() == 3


@pytest.mark.parametrize("test_input, test_label, expected", [(0.0,0.5,0.5), (0.1,0.2,0.1), (0.1,0.55,0.45), (0.1,0.85,0.25), 
(0.1,0.9,0.2), (0.4,0.89,0.49),(0.4,0.91,0.49), (0.75,0.85,0.1), (0.9,0.99,0.09), (0.903,0.270,0.367), (-1.0,2.0,2.0)])
def test_single_custom_loss_function2(test_input, test_label, expected):
    assert single_custom_loss_function(test_input, test_label) == pytest.approx(expected, 0.000001)
    assert single_custom_loss_function(test_label, test_input) == pytest.approx(expected, 0.000001)