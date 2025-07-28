# ================================
# Four Plots
# Displaying four plots: PSD, Spectrogram, FFT, STFT
# ================================

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

# ------------------------------------------------------------
# 1. Load EEG Data
# ------------------------------------------------------------
print("Loading EEG data...")
raw = mne.io.read_raw_eeglab(
    r'derivatives\sub-005\eeg\sub-005_task-eyesclosed_eeg.set',
    preload=True
)
data = raw.get_data()        # (n_channels, n_timepoints)
sfreq = raw.info['sfreq']    # 500 Hz
print(f"Data shape: {data.shape}, Sampling frequency: {sfreq} Hz")

# Select first channel for analysis
channel_idx = 0
signal = data[channel_idx]  # Signal from first channel
print(f"Analyzing channel {channel_idx+1}")

# ------------------------------------------------------------
# 2. Select Time Window
# ------------------------------------------------------------
# Select first 4 seconds of data for analysis
start_time = 0  # Start from 0 seconds
duration = 4   # Analyze 4 seconds
start_sample = int(start_time * sfreq)
end_sample = int((start_time + duration) * sfreq)
signal_segment = signal[start_sample:end_sample]  

print(f"Selected {duration}s segment: {signal_segment.shape}")

# ------------------------------------------------------------
# 3. Compute Various Spectral Analyses
# ------------------------------------------------------------
print("Computing spectral analysis...")

# 3.1 Compute FFT using scipy.fft
N = len(signal_segment)
# Use scipy.fft instead of numpy.fft
from scipy import fft
fft_result = fft.fft(signal_segment)
fft_freqs = fft.fftfreq(N, 1/sfreq)
# Get only positive frequencies (first half)
positive_freq_mask = fft_freqs >= 0
fft_freqs = fft_freqs[positive_freq_mask]
fft_result = fft_result[positive_freq_mask]
# Correct FFT amplitude: divide by N to get correct units
fft_amp = np.abs(fft_result) / N

# 3.2 Compute PSD (Power Spectral Density)
freqs_psd, psd = scipy_signal.welch(
    signal_segment, 
    fs=sfreq, 
    nperseg=128, 
    noverlap=64,
    window='hann'
)

# 3.3 Compute STFT Spectrogram
n_fft = 128
hop_length = 64
freqs_stft, times_stft, Sxx = scipy_signal.spectrogram(
    signal_segment,
    fs=sfreq,
    nperseg=n_fft,
    noverlap=n_fft-hop_length,
    window='hann',
    scaling='density'
)

# 3.4 Compute STFT (using scipy.signal.stft)
freqs_scipy, times_scipy, Zxx = scipy_signal.stft(
    signal_segment,
    fs=sfreq,
    nperseg=128,
    noverlap=64,
    window='hann'
)
# Compute STFT magnitude
Sxx_scipy = np.abs(Zxx)

# ------------------------------------------------------------
# 4. Plot Four Figures
# ------------------------------------------------------------
print("Plotting four analysis figures...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('EEG Spectral Analysis - PSD, Spectrogram, FFT (scipy.fft), STFT', fontsize=16)

# 4.1 PSD Plot
axes[0, 0].semilogy(freqs_psd, psd, 'b-', linewidth=2)
axes[0, 0].set_title('Power Spectral Density (PSD)')
axes[0, 0].set_xlabel('Frequency (Hz)')
axes[0, 0].set_ylabel('Power Spectral Density (V²/Hz)')
axes[0, 0].set_xlim(0, 50)
axes[0, 0].grid(True)

# 4.2 Spectrogram Plot
im1 = axes[0, 1].pcolormesh(times_stft, freqs_stft, Sxx, 
                             shading='gouraud', cmap='jet')
axes[0, 1].set_title('STFT Spectrogram')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Frequency (Hz)')
axes[0, 1].set_ylim(0, 50)
plt.colorbar(im1, ax=axes[0, 1], label='Power Spectral Density')

# 4.3 FFT Plot - Using scipy.fft
axes[1, 0].plot(fft_freqs, fft_amp, 'r-', linewidth=2)
axes[1, 0].set_title('FFT Amplitude Spectrum (scipy.fft)')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Amplitude (V)')
axes[1, 0].set_xlim(0, 50)
axes[1, 0].grid(True, alpha=0.3)

# 4.4 STFT (scipy.signal.stft) Plot
im2 = axes[1, 1].pcolormesh(times_scipy, freqs_scipy, Sxx_scipy, 
                             shading='gouraud', cmap='viridis')
axes[1, 1].set_title('STFT (scipy.signal.stft)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Frequency (Hz)')
axes[1, 1].set_ylim(0, 50)
plt.colorbar(im2, ax=axes[1, 1], label='Magnitude')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. Print Statistical Information
# ------------------------------------------------------------
print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)
print(f"Signal length: {len(signal_segment)} samples ({duration}s)")
print(f"Sampling frequency: {sfreq} Hz")
print(f"Frequency resolution (FFT): {sfreq/N:.2f} Hz")
print(f"Frequency resolution (PSD): {sfreq/256:.2f} Hz")
print(f"Time resolution (Spectrogram): {hop_length/sfreq:.3f} s")
print(f"Frequency range: 0 - {sfreq/2:.1f} Hz")

# Find dominant frequency component
max_freq_idx = np.argmax(psd)
max_freq = freqs_psd[max_freq_idx]
print(f"Dominant frequency: {max_freq:.1f} Hz")

print("="*50)
print("Analysis completed!") 