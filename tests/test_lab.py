import numpy as np
import pytest
import torch
import torch.nn as nn

from cv import lab
from cv.segmentation import TinyUNet


@pytest.mark.parametrize("w,k,p,s", [(256, 3, 1, 1), (256, 3, 0, 2), (31, 5, 2, 2), (64, 1, 0, 1)])
def test_conv_output_size_matches_torch(w, k, p, s):
    out = nn.Conv2d(1, 1, k, stride=s, padding=p)(torch.zeros(1, 1, w, w))
    assert out.shape[-1] == lab.conv_output_size(w, k, p, s)


def test_conv_layer_params():
    conv = nn.Conv2d(16, 32, 3)
    assert lab.conv_layer_stats(16, 32, 3, 64, 64)["params"] == sum(p.numel() for p in conv.parameters())


def test_unet_table_total():
    model = TinyUNet(16)
    rows = lab.unet_layer_table(model)
    assert sum(r["params"] for r in rows) == sum(p.numel() for p in model.parameters())


def test_adam_reduces_bowl_loss():
    path = lab.optimizer_path("Adam", "bowl", 0.1, 100, (-4.0, 2.0))
    assert lab.loss_value("bowl", path[-1]) < lab.loss_value("bowl", path[0])


def test_relu_derivative():
    _, _, grad = lab.activation_curves("ReLU", x=[2.0, -2.0])
    assert np.allclose(grad, [1.0, 0.0])
