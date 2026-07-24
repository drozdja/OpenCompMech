"""Regression tests for the nominal classifier-free-guidance fast path."""

import unittest

import torch

from src.ml.guided_sample import guided_sample


class _CountingZeroModel:
    def __init__(self):
        self.calls = []

    def __call__(self, x, _t, cond, scal):
        self.calls.append((cond.detach().clone(), scal.detach().clone()))
        return torch.zeros_like(x)


def _sample(model, cfg_scale):
    cond = torch.full((1, 11, 64, 64), 0.25)
    scal = torch.full((1, 15), 0.5)
    return guided_sample(model, None, cond, scal, None, None, None, None, 0.0,
                         steps=1, device="cpu", cfg_scale=cfg_scale,
                         target_vf=None, domain=None)


class GuidedSampleCfgFastPathTest(unittest.TestCase):
    def test_nominal_cfg_is_direct_conditional_sampling(self):
        torch.manual_seed(20260721)
        unconditional_disabled = _CountingZeroModel()
        nominal = _sample(unconditional_disabled, 1.0)
        self.assertEqual(len(unconditional_disabled.calls), 1)
        self.assertTrue(torch.all(unconditional_disabled.calls[0][0] == 0.25))
        self.assertTrue(torch.all(unconditional_disabled.calls[0][1] == 0.5))

        # cfg=None has always meant direct conditional sampling.  With the same
        # seed, the nominal CFG identity now follows that exact branch.
        torch.manual_seed(20260721)
        plain_model = _CountingZeroModel()
        plain = _sample(plain_model, None)
        self.assertEqual(len(plain_model.calls), 1)
        self.assertTrue(torch.equal(nominal, plain))

    def test_non_nominal_cfg_keeps_both_branches(self):
        model = _CountingZeroModel()
        _sample(model, 0.5)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(torch.all(model.calls[0][0] == 0.25))
        self.assertTrue(torch.all(model.calls[1][0] == 0.0))


if __name__ == "__main__":
    unittest.main()
