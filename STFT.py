'''
(B, C, T) 
    ↓ view()
(B*C, T) 
    ↓ torch.stft()
(B*C, F, T_new, 2)  
    ↓ torch.norm()
(B*C, F, T_new)    
    ↓ view()
(B, C, F, T_new)    
    ↓ only preserving F_0_50 (0<F<50Hz)
(B, C, F_0_50, T_new)  
    ↓ reshape to Transformer format
(B, C*F_0_50, T_new)  
'''

import mne
import torch
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load EEG Data
# ------------------------------------------------------------
raw = mne.io.read_raw_eeglab(
    r'derivatives\sub-001\eeg\sub-001_task-eyesclosed_eeg.set',
    preload=True
)
data  = raw.get_data()        # (n_channels, n_timepoints)
sfreq = raw.info['sfreq']     # 500 Hz

# ------------------------------------------------------------
# 2. Sliding Window Epoch Segmentation (numpy -> torch)
# ------------------------------------------------------------
epoch_len   = 4               # seconds
overlap_len = 2               # seconds
n_per_seg   = int(sfreq * epoch_len)     # 2000 samples
n_overlap   = int(sfreq * overlap_len)   # 1000 samples
step        = n_per_seg - n_overlap      # 1000 samples

# Generate start indices for sliding windows
starts = np.arange(0, data.shape[1] - n_per_seg + 1, step)

# Extract epochs using advanced indexing: (n_epochs, n_channels, n_timepoints)
epochs_np = np.stack([data[:, s:s+n_per_seg] for s in starts], axis=0)
epochs_np = epochs_np.astype(np.float32)          # (n_epochs=298, 19, 2000)

# Convert to torch tensor: shape (B, C, T)
epochs = torch.from_numpy(epochs_np)               # No need for .cuda(), CPU works fine
print("Epochs tensor shape:", epochs.shape)        # (n_epochs=298, 19, 2000)

# ------------------------------------------------------------
# 3. Batch STFT Processing
#    Dimension transformation: (B,C,T) -> (B*C,T) -> STFT -> (B*C,F,T_new,2) -> (B,C,F,T_new) -> (B,T_new,C*F)
# ------------------------------------------------------------
B, C, T = epochs.shape  # B=batch_size, C=channels=19, T=timepoints
n_fft      = 128
hop_length = 64

# Flatten to 2D: (B*C, T)
# This allows torch.stft to process all channels in parallel
x_flat = epochs.view(B * C, T)

# torch.stft returns (N, F, T_new, 2) where last dimension is (real, imaginary)
stft_complex = torch.stft(x_flat,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=n_fft,
                          window=torch.hann_window(n_fft),
                          onesided=True,
                          return_complex=False)   # (B*C, F, T_new, 2)

'''
The key advantage of torch.stft is that it treats the last dimension as time
and all other dimensions as batch dimensions, performing FFT-convolution in parallel.
NumPy doesn't have such high-dimensional parallel interface, requiring manual loops.
'''
F      = stft_complex.size(1)  # Number of frequency bins
T_new  = stft_complex.size(2)  # Number of time windows

# Extract magnitude: sqrt(real^2 + imag^2)
stft_amp = torch.norm(stft_complex, dim=-1)        # (B*C, F, T_new)

# Optional: logarithmic compression to avoid large values
# stft_amp = torch.log1p(stft_amp)

# Restore original dimensions: (B, C, F, T_new)
stft_amp = stft_amp.view(B, C, F, T_new)

# Frequency filtering: keep only 0-50 Hz (excluding 0 Hz)
freqs = torch.fft.rfftfreq(n_fft, 1/sfreq)
freq_mask = (freqs > 0) & (freqs <= 50)  # Exclude 0 Hz
stft_amp = stft_amp[:, :, freq_mask, :]  # (B, C, F_0_50, T_new)
F_0_50 = freq_mask.sum().item()
print(f"Number of frequency bins (0-50 Hz): {F_0_50}")
stft_amp = stft_amp.view(B, C, F_0_50, T_new)

# Reshape for Transformer: (B, T_new, C*F)
# This format is more suitable for attention mechanisms
stft_final = stft_amp.permute(0, 3, 1, 2).contiguous()  # (B, T_new, C, F_0_50)
stft_final = stft_final.view(B, T_new, C * F_0_50)      # (B, T_new, C*F_0_50)
stft_final = stft_final.permute(0, 2, 1)                # (B, C*F_0_50, T_new)

# Note: permute changes memory layout, so contiguous() is needed before view()
print("STFT final shape:", stft_final.shape)       # e.g. (B=298, C*F_0_50=627, T_new=63)

# ------------------------------------------------------------
# 4. Visualize STFT for First Channel, First Epoch
# ------------------------------------------------------------
amp0 = stft_final[0].cpu().numpy()  # (C*F_0_50, T_new)
amp0_db = 20 * np.log10(amp0 + 1e-6)
times = torch.arange(T_new) * hop_length / sfreq

plt.figure(figsize=(6, 4))
plt.pcolormesh(times, np.arange(C*F_0_50), amp0_db, shading='gouraud', cmap='jet')
plt.colorbar(label='Amplitude (dB)')
plt.title('STFT (torch) - First electrode, first epoch')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()



