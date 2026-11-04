"""CPU checks of the paper target, locality, saved-args loading and gradients.

Run from the repository root: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np
import torch

from eeg_lejepa_v3 import EEGLeJEPA_v3, LinearPatchTokenizer, v3_kwargs_from_args


def paper_args():
    return dict(d_model=32, encoder_layers=1, patch_len=200, n_bands=5,
                cf_d_band=8, tokenizer_type="linear", mask_axis="time",
                target_type="fft_bandpower", fft_bands="standard",
                fft_taper="rect", target_log="log", jepa_weight=0.0,
                cf_weight=1.0, sigreg_lambda=0.0)


class SpectralObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        torch.manual_seed(17)
        return EEGLeJEPA_v3(**v3_kwargs_from_args(paper_args()))

    def test_power_of_bin_centered_sinusoids_and_phase(self):
        m = self.model().eval()
        n = torch.arange(200, dtype=torch.float32)
        # One positive DFT bin in each classical band, with integer cycles.
        for band, k in enumerate((2, 5, 8, 16, 30)):
            for phase in (0.0, 0.71, 2.2):
                x = 2.0 * torch.cos(2 * torch.pi * k * n / 200 + phase)
                logp = m._fft_band_logpow(x[None, :, None])[0, 0, 0]
                power = logp.exp() - m.log_floor
                self.assertAlmostEqual(float(power[band]), 2.0, places=4)
                other = torch.cat((power[:band], power[band + 1:]))
                self.assertLess(float(other.abs().max()), 2e-5)

    def test_numpy_reference_and_zero_floor(self):
        m = self.model().eval()
        rng = np.random.default_rng(12)
        x = rng.normal(size=(2, 600, 3)).astype(np.float32)
        patches = x.transpose(0, 2, 1).reshape(2, 3, 3, 200)
        spectrum = np.abs(np.fft.rfft(patches, axis=-1)) ** 2
        freqs = np.fft.rfftfreq(200, 1 / 256)
        expected = np.stack([2 * spectrum[..., (freqs >= lo) & (freqs < hi)].sum(-1)
                             / 200 ** 2 for lo, hi in m.fft_edges], axis=-1)
        actual = m._fft_band_logpow(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(actual, np.log(expected + 1e-3), rtol=2e-6, atol=2e-6)
        silent = m._fft_band_logpow(torch.zeros(1, 200, 1))
        self.assertTrue(torch.isfinite(silent).all())
        torch.testing.assert_close(silent, torch.full_like(silent, np.log(1e-3)))

    def test_target_and_visible_linear_embedding_are_patch_local(self):
        m = self.model().eval()
        x = torch.randn(1, 600, 2)
        y = x.clone()
        y[:, 200:400] += 100 * torch.randn_like(y[:, 200:400])
        torch.testing.assert_close(m._fft_band_logpow(x)[:, :, 0],
                                   m._fft_band_logpow(y)[:, :, 0])
        ex, _, _ = m.tokenizer(x)
        ey, _, _ = m.tokenizer(y)
        torch.testing.assert_close(ex[:, :, 0], ey[:, :, 0])

    def test_checkpoint_configuration_forward_and_backward(self):
        m = self.model()
        clone = EEGLeJEPA_v3(**v3_kwargs_from_args(paper_args()))
        clone.load_state_dict(m.state_dict(), strict=True)
        self.assertIsInstance(clone.tokenizer, LinearPatchTokenizer)
        x = torch.randn(2, 600, 3)
        m.train()
        torch.manual_seed(123)
        loss = m(x)
        self.assertTrue(torch.isfinite(loss["total"]))
        torch.testing.assert_close(loss["total"].detach(), loss["cf"])
        loss["total"].backward()
        self.assertGreater(float(m.tokenizer.patch_proj.weight.grad.abs().sum()), 0)
        m.eval()
        with torch.no_grad():
            features = m._encode(m._tokenize(x)).mean(1)
        self.assertEqual(tuple(features.shape), (2, 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
