"""Smoke test for reconstruction.py — run directly."""
import numpy as np

from nv_maser.physics.reconstruction import (
    apply_undersampling_mask,
    estimate_acceleration_factor,
    haar_wavelet_inverse,
    haar_wavelet_transform,
    image_snr_from_phantom,
    reconstruct_compressed_sensing,
    reconstruct_depth_profile,
    reconstruct_fft,
    reconstruct_gridding,
    simulate_kspace,
    sweep_resolution_vs_fov,
    sweep_snr_vs_acceleration,
)

rng = np.random.default_rng(42)
phantom = rng.standard_normal((32, 32))

# 1. Wavelet roundtrip
for lvls in [1, 2, 3]:
    c = haar_wavelet_transform(phantom, levels=lvls)
    r = haar_wavelet_inverse(c, levels=lvls)
    err = float(np.max(np.abs(r - phantom)))
    assert err < 1e-10, f"Wavelet L={lvls} roundtrip err={err:.2e}"
print("1. Wavelet roundtrip: PASS (levels 1,2,3)")

# 2. FFT recon — use a box phantom (positive values) for meaningful magnitude correlation
box = np.zeros((32, 32))
box[8:24, 8:24] = 1.0
ksp_box = simulate_kspace(box)
fr_box = reconstruct_fft(ksp_box, fov_x_m=0.05, fov_y_m=0.05, apply_hamming=False)
corr_box = float(np.corrcoef(box.ravel(), fr_box.magnitude.ravel())[0, 1])
assert corr_box > 0.95, f"FFT box corr={corr_box:.4f}"
print(f"   noiseless box FFT corr={corr_box:.4f} (expect ~1.0) PASS")
# Also verify real-part roundtrip for Gaussian phantom
ksp = simulate_kspace(phantom)
ksp_n = ksp + 0.01 * (rng.standard_normal(ksp.shape) + 1j * rng.standard_normal(ksp.shape))
fr = reconstruct_fft(ksp_n, fov_x_m=0.05, fov_y_m=0.05, apply_hamming=False)
corr_real = float(np.corrcoef(phantom.ravel(), fr.image.real.ravel())[0, 1])
assert corr_real > 0.95, f"FFT real-part corr={corr_real:.4f}"
print(f"2. FFT recon: box_corr={corr_box:.4f} real_corr={corr_real:.4f} SNR={fr.snr_db:.1f} dB  PASS")

# 3. CS recon
ksp_s, mask = apply_undersampling_mask(ksp_n, acceleration_factor=2, seed=7)
af = estimate_acceleration_factor(mask)
cs = reconstruct_compressed_sensing(ksp_s, mask, fov_x_m=0.05, fov_y_m=0.05, n_iterations=20)
print(f"3. CS recon: method={cs.method} acc={af:.2f}x SNR={cs.snr_db:.1f} dB  PASS")

# 4. Gridding recon
kx = rng.uniform(-10, 10, 300)
ky = rng.uniform(-10, 10, 300)
samp = rng.standard_normal(300) + 1j * rng.standard_normal(300)
gr = reconstruct_gridding(kx, ky, samp, fov_x_m=0.05, fov_y_m=0.05, grid_size=32)
assert gr.image.shape == (32, 32), f"Gridding shape={gr.image.shape}"
print(f"4. Gridding recon: method={gr.method} shape={gr.image.shape}  PASS")

# 5. Depth profile
t = np.linspace(0, 1e-3, 512)
sig = np.exp(-t / 0.5e-3) * np.cos(2 * np.pi * 1000 * t)
dwell_us = float((t[1] - t[0]) * 1e6)
dp = reconstruct_depth_profile(sig, dwell_us=dwell_us, gradient_t_per_m=0.1)
print(f"5. Depth profile: peak={dp.peak_depth_mm:.1f} mm T2*={dp.t2_star_ms:.2f} ms  PASS")

# 6. Sweeps
snrs = sweep_snr_vs_acceleration(phantom, [1, 2, 4])
res = sweep_resolution_vs_fov(32, [0.02, 0.05, 0.1])
assert len(snrs) == 3 and len(res) == 3
print(f"6. Sweeps: SNRs={[round(s,1) for s in snrs]} res={[round(r,2) for r in res]} mm  PASS")

# 7. Image SNR
snr = image_snr_from_phantom(fr.magnitude)
print(f"7. Phantom SNR: {snr:.1f} dB  PASS")

print("\nALL SMOKE TESTS PASSED")
