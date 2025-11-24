import numpy as np

import fmri_fm_eval.nisc as nisc


def test_parcel_average():
    x = np.random.randn(10, nisc.FSLR64K_NUM_VERTICES).astype(np.float32)

    parcavg = nisc.parcel_average_schaefer_fslr64k(400, sparse=False)
    z = parcavg(x)
    assert z.shape == (10, 400)

    idx = 9
    assert np.allclose(z[:, idx], x[:, parcavg.parc == (idx + 1)].mean(axis=1))


def test_parcel_average_sparse():
    x = np.random.randn(10, nisc.FSLR64K_NUM_VERTICES).astype(np.float32)

    parcavg_sprs = nisc.parcel_average_schaefer_fslr64k(400, sparse=True)
    z_sprs = parcavg_sprs(x)

    parcavg = nisc.parcel_average_schaefer_fslr64k(400, sparse=False)
    z = parcavg(x)
    assert np.allclose(z, z_sprs, atol=1e-5)


def test_flat_resampler():
    resampler = nisc.flat_resampler_fslr64k_224_560()
    assert resampler.mask_.shape == (224, 560)
    assert resampler.mask_.sum() == 77763

    x = np.random.randn(nisc.FSLR64K_NUM_VERTICES).astype(np.float32)
    x_flat = resampler.transform(x, interpolation="nearest")
    x_ = resampler.inverse(x_flat)

    # nb, even with nearest interpolation and restricting to vertices used in the
    # forward mapping, the inverse is not perfect. Bc the nearest neighbor of your
    # nearest neighbor is not always you.
    #
    #   .........x...x.........x
    #   ..........y......y......
    #
    v = x[resampler.point_mask_][resampler._neigh_ind]
    v_ = x_[resampler.point_mask_][resampler._neigh_ind]
    assert np.mean(v == v_) > 0.97
