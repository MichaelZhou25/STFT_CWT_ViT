import mne
import numpy as np
import pywt
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load EEG Data and Sliding Window Segmentation
# ------------------------------------------------------------
raw = mne.io.read_raw_eeglab(
    r'derivatives\sub-087\eeg\sub-087_task-eyesclosed_eeg.set',
    preload=True
)
data = raw.get_data()        # (n_channels, n_timepoints)
sfreq = raw.info['sfreq']    # Usually 500 Hz

# Sliding window segmentation
epoch_len = 4     # seconds
overlap_len = 2   # seconds
n_per_seg = int(sfreq * epoch_len)
n_overlap = int(sfreq * overlap_len)
step = n_per_seg - n_overlap
starts = np.arange(0, data.shape[1] - n_per_seg + 1, step)
epochs_np = np.stack([data[:, s:s+n_per_seg] for s in starts], axis=0).astype(np.float32)
# shape: (B, C, T)
B, C, T = epochs_np.shape

print("Epochs shape:", epochs_np.shape)

# ------------------------------------------------------------
# 2. Batch CWT Transform, Adapted for Transformer Format
# ------------------------------------------------------------
mother_wavelet = 'cmor1.5-1.0'   # Complex Morlet (commonly used for analysis)
scale_min, scale_max = 2, 64
scales = np.arange(scale_min, scale_max + 1)   # Number of scales determines frequency resolution

flat = epochs_np.reshape(B * C, T)  # (B*C, T)
cwt_coeffs_list = []
for sig in flat:
    coefs, freqs = pywt.cwt(sig, scales, mother_wavelet, sampling_period=1/sfreq)
    cwt_coeffs_list.append(np.abs(coefs))   # Take magnitude as input
cwt_coeffs = np.stack(cwt_coeffs_list, axis=0)  # (B*C, F, T)
cwt_coeffs = cwt_coeffs.reshape(B, C, len(scales), T)  # (B, C, F, T)

# Keep only 0<F<=50Hz
freq_mask = (freqs > 0) & (freqs <= 50)
cwt_coeffs = cwt_coeffs[:, :, freq_mask, :]    # (B, C, F_0_50, T)
F_0_50 = freq_mask.sum()
print(f"Number of frequency bins (0-50Hz): {F_0_50}")

# Transformer input shape: (B, C*F_0_50, T)
cwt_final = cwt_coeffs.reshape(B, C * F_0_50, T)
print("CWT final shape:", cwt_final.shape)      # (B, C*F_0_50, T)

# ------------------------------------------------------------
# 3. Visualize CWT Scalogram (First Epoch, First Channel)
# ------------------------------------------------------------
# (B, C, F_0_50, T)
amp0 = cwt_coeffs[0, 0]      # shape: (F_0_50, T)
amp0_db = 20 * np.log10(amp0 + 1e-6)
times = np.arange(T) / sfreq

plt.figure(figsize=(6, 4))
plt.pcolormesh(times, freqs[freq_mask], amp0_db, shading='gouraud', cmap='jet')
plt.colorbar(label='Amplitude (dB)')
plt.title('CWT - First epoch, first channel')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
