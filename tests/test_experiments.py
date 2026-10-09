import numpy as np
import pytest
import torch
import torch.nn as nn

from cv import experiments as ex
from cv import segmentation as seg
from cv.models import MODELS_DIR


@pytest.fixture(scope="module")
def unet():
    torch.manual_seed(0)
    return seg.TinyUNet(4).eval()


def stain(share: float = 0.1, size: int = 100) -> np.ndarray:
    truth = np.zeros((size, size), np.uint8)
    truth[: int(size * share)] = 1
    return truth


# losses and metrics

def test_perfect_mask_scores():
    truth = stain()
    r = ex.mask_scores(truth.astype(np.float32), truth)
    assert r["pixel accuracy"] == r["IoU"] == r["Dice"] == 1.0
    assert r["BCE"] == pytest.approx(-np.log(1 - ex.BCE_EPS))


def test_empty_mask_scores_high_accuracy_and_zero_overlap():
    r = ex.mask_scores(np.zeros((100, 100)), stain(0.1))
    assert r["pixel accuracy"] == pytest.approx(0.9)
    assert r["IoU"] == 0 and r["Dice"] == 0


def test_dice_from_iou():
    rng = np.random.default_rng(0)
    r = ex.mask_scores(rng.random((64, 64)), rng.random((64, 64)) > 0.7)
    assert r["Dice"] == pytest.approx(2 * r["IoU"] / (1 + r["IoU"]))


def test_baseline_masks():
    truth = np.zeros((50, 50), np.uint8)
    truth[10:20, 5:30] = 1
    masks = ex.baseline_masks(truth)
    assert masks["empty mask"].sum() == 0 and masks["all stain"].sum() == 2500
    assert masks["filled box"].sum() == truth.sum()  # the stain is a rectangle here


def test_grow_mask_grows_and_shrinks():
    truth = np.zeros((40, 40), np.uint8)
    truth[10:30, 10:30] = 1
    assert ex.grow_mask(truth, 0).sum() == truth.sum()
    assert ex.grow_mask(truth, 3).sum() > truth.sum() > ex.grow_mask(truth, -3).sum()
    assert ex.grow_mask(truth, -3)[20, 20] == 1 and ex.grow_mask(truth, 3)[8, 20] == 1


def test_clean_mask_off_is_unchanged_and_opening_removes_specks():
    mask = np.zeros((60, 60), bool)
    mask[10:40, 10:40] = True
    mask[50, 50] = True                                  # a one-pixel speck
    assert (seg.clean_mask(mask, 0, 0) == mask).all()
    opened = seg.clean_mask(mask, 3, 0)
    assert not opened[50, 50] and opened[25, 25]
    holed = mask.copy()
    holed[25, 25] = False                                # a one-pixel gap
    assert seg.clean_mask(holed, 0, 3)[25, 25]


# pruning

def test_prune_copy_hits_the_amount_and_keeps_the_original(unet):
    before = ex.sparsity(unet)
    pruned = ex.prune_copy(unet, 0.5)
    zeros, total = ex.sparsity(pruned)
    assert before[0] == 0 and total == before[1]
    assert zeros / total == pytest.approx(0.5, abs=1e-3)
    assert {k: v.shape for k, v in pruned.state_dict().items()} == {k: v.shape for k, v in unet.state_dict().items()}


def test_pruning_keeps_the_file_size(unet):
    assert seg.state_size_mb(ex.prune_copy(unet, 0.9)) == pytest.approx(seg.state_size_mb(unet), rel=0.01)


def test_prune_zero_is_a_plain_copy(unet):
    copy = ex.prune_copy(unet, 0.0)
    assert copy is not unet and ex.sparsity(copy)[0] == 0


def test_median_ms_is_positive():
    assert ex.median_ms(lambda: sum(range(1000)), runs=3) > 0


def test_measure_and_pruning_rows(unet):
    img, truth = seg.make_scene(28, size=64)
    val = seg.make_dataset(4, 1)
    rows = ex.pruning_rows(unet, [0.0, 0.5], img, truth, val, runs=2)
    assert [r["sparsity"] for r in rows] == pytest.approx([0.0, 0.5], abs=1e-3)
    assert rows[1]["non-zero weights"] < rows[0]["non-zero weights"]
    assert all(0 <= r["IoU this wall"] <= 1 and r["latency ms"] > 0 for r in rows)


# normalization and init

def test_normalize_input_modes():
    img = np.full((8, 8, 3), 255, np.uint8)
    div, raw, imnet = (ex.normalize_input(img, m) for m in ex.INPUT_MODES)
    assert div.max() == 1 and raw.max() == 255
    assert imnet[0, 0, 0] == pytest.approx((1 - ex.IMAGENET_MEAN[0]) / ex.IMAGENET_STD[0])


def test_enc1_stats_shapes(unet):
    stats = ex.enc1_stats(unet, torch.rand(3, 32, 32))
    assert all(v.shape == (4,) for v in stats.values())


def test_init_zeros_gives_zero():
    assert ex.init_stds("zeros", False) == [0.0] * 12


def test_init_normal_explodes_and_default_shrinks():
    assert ex.init_stds("N(0, 1)", False)[-1] > 10
    assert ex.init_stds("PyTorch default", False)[-1] < 0.1
    assert 0.1 <= ex.init_stds("Kaiming", False)[-1] <= 10


def test_batchnorm_rescues_any_init():
    for scheme in ["N(0, 1)", "PyTorch default", "Kaiming"]:
        assert 0.1 <= ex.init_stds(scheme, True)[-1] <= 10


# architectures

class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.bn = nn.BatchNorm2d(8)
        self.act = nn.SiLU()
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        y = self.act(self.bn(self.dw(self.conv(x))))
        return self.fc(torch.cat([y.mean((2, 3)), y.amax((2, 3))], 1))


def test_inspect_small_model():
    info = ex.inspect_model(Small())
    assert info["conv layers"] == 2 and info["depthwise convs"] == 1 and info["BatchNorm layers"] == 1
    assert info["activations"] == {"SiLU": 1} and info["params"] == sum(p.numel() for p in Small().parameters())


def test_inspect_unet(unet):
    info = ex.inspect_model(unet)
    assert info["conv layers"] == 11 and info["BatchNorm layers"] == 10 and info["activations"] == {"ReLU": 10}
    assert info["depthwise convs"] == info["attention blocks"] == info["residual bottlenecks"] == 0


@pytest.mark.parametrize("model", [Small(), seg.TinyUNet(4)])
def test_params_by_category_sums_to_total(model):
    cats = ex.params_by_category(model)
    assert sum(cats.values()) == sum(p.numel() for p in model.parameters())


def test_params_by_category_small():
    cats = ex.params_by_category(Small())
    assert cats["depthwise conv"] == 8 * 9 + 8 and cats["linear"] == 34 and cats["BatchNorm"] == 16


def test_count_cats(unet):
    assert ex.count_cats(unet, 32) == 2
    assert ex.count_cats(Small(), 8) == 1


def test_unet_with_both_skips_matches_predict(unet):
    img, _ = seg.make_scene(28, size=64)
    assert np.allclose(ex.unet_without_skips(unet, img, True, True), seg.predict(unet, img), atol=1e-6)


def test_dropping_a_skip_changes_the_output(unet):
    img, _ = seg.make_scene(28, size=64)
    assert not np.allclose(ex.unet_without_skips(unet, img, False, True), seg.predict(unet, img))


# mixup and cutmix

def test_square_crop():
    assert ex.square(np.zeros((300, 500, 3), np.uint8)).shape == (224, 224, 3)


def test_mixup_endpoints_and_middle():
    a, b = np.full((4, 4, 3), 200, np.uint8), np.zeros((4, 4, 3), np.uint8)
    assert (ex.mixup(a, b, 1.0) == a).all() and (ex.mixup(a, b, 0.0) == b).all()
    assert (ex.mixup(a, b, 0.25) == 50).all()


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.7, 1.0])
def test_cutmix_share_matches_pixels(lam):
    a, b = np.ones((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8)
    out, share = ex.cutmix(a, b, lam)
    assert out[..., 0].mean() == pytest.approx(share)
    assert share == pytest.approx(lam, abs=0.03)


# linear probe

def test_probe_images_are_balanced():
    imgs, labels = ex.probe_images(3, 0)
    assert len(imgs) == 6 and labels.tolist() == [1, 1, 1, 0, 0, 0]
    assert imgs[0].shape == (seg.TRAIN_SIZE, seg.TRAIN_SIZE, 3)


def test_train_probe_separates_easy_features():
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 20)
    feats = rng.normal(0, 1, (40, 1280)).astype(np.float32) + labels[:, None] * 2.0
    probe = ex.train_probe(feats, labels, steps=100)
    assert sum(p.numel() for p in probe.parameters()) == 2562
    assert ex.probe_accuracy(probe, feats, labels) == 1.0


@pytest.mark.skipif(not (MODELS_DIR / "yolo26n-cls.pt").exists(), reason="run setup_models.py")
def test_backbone_features_shape():
    from cv.models import load_yolo
    imgs, _ = ex.probe_images(2, 0)
    assert ex.backbone_features(load_yolo("yolo26n-cls.pt").model, imgs).shape == (4, 1280)
