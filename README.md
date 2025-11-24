# EXP-1(C) ANALYSIS OF AUDIO SIGNAL FOR NOISE REMOVAL
# AIM:
# To analyze an audio sgnal and remove noise

# APPARATUS REQUIRED:
PC installed with SCILAB.

# PROGRAM:
```
# ================================
#   AUDIO NOISE REMOVAL & SEPARATION
# ================================

# Step 1: Install packages
!pip install -q librosa noisereduce soundfile

# Step 2: Upload clean and noise recordings
from google.colab import files
print("Upload clean/normal audio (speech/music)")
uploaded = files.upload()
clean_file = next(iter(uploaded.keys()))

print("Upload noise-only audio (background)")
uploaded = files.upload()
noise_file = next(iter(uploaded.keys()))

# Step 3: Load audios
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import noisereduce as nr

clean, sr_c = librosa.load(clean_file, sr=None, mono=True)
noise, sr_n = librosa.load(noise_file, sr=None, mono=True)

# Resample noise if sample rates differ
if sr_c != sr_n:
    print(f"Resampling noise from {sr_n} Hz → {sr_c} Hz")
    noise = librosa.resample(noise, orig_sr=sr_n, target_sr=sr_c)
    sr_n = sr_c

sr = sr_c
print(f"Clean audio SR = {sr_c}, Noise audio SR = {sr_n}")
print(f"Clean length = {len(clean)/sr:.2f} sec, Noise length = {len(noise)/sr:.2f} sec")

# Step 4: Make lengths equal (pad or cut noise)
if len(noise) < len(clean):
    reps = int(np.ceil(len(clean)/len(noise)))
    noise = np.tile(noise, reps)[:len(clean)]
else:
    noise = noise[:len(clean)]

# Step 5: Create noisy mixture
noisy = clean + noise * 0.5        # adjust noise scaling factor
print("Generated noisy signal.")

# Step 6: Play audio
print("\n--- Original Clean Audio ---")
display(Audio(clean, rate=sr))

print("\n--- Noise Sample ---")
display(Audio(noise, rate=sr))

print("\n--- Noisy (Merged) Audio ---")
display(Audio(noisy, rate=sr))

# Step 7: Frequency Analysis (FFT Spectra)
def plot_spectrum(signal, sr, title):
    n_fft = 2 * 14
    Y = np.fft.rfft(signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    magnitude = np.abs(Y)

    plt.figure(figsize=(12,4))
    plt.semilogy(freqs, magnitude+1e-12)
    plt.xlim(0, sr/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (log)")
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_spectrum(clean, sr, "Spectrum of Clean Audio")
plot_spectrum(noise, sr, "Spectrum of Noise")
plot_spectrum(noisy, sr, "Spectrum of Noisy Audio")

# Step 8: Noise Reduction
reduced = nr.reduce_noise(y=noisy, y_noise=noise, sr=sr)

# Step 9: Extract estimated noise
estimated_noise = noisy - reduced

print("\n--- Denoised / Cleaned Audio ---")
display(Audio(reduced, rate=sr))

print("\n--- Extracted Noise Component ---")
display(Audio(estimated_noise, rate=sr))

# Step 10: Spectrograms
def plot_spec(signal, sr, title):
    D = librosa.stft(signal, n_fft=1024, hop_length=512)
    S_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(12,5))
    librosa.display.specshow(S_dB, sr=sr, hop_length=512,
                             x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.ylim(0, sr/2)
    plt.show()

plot_spec(noisy, sr, "Spectrogram of Noisy Audio")
plot_spec(reduced, sr, "Spectrogram of Denoised Audio")
plot_spec(estimated_noise, sr, "Spectrogram of Extracted Noise")

```
# ORIGINAL CLEAN AUDIO:
download.wav

# NOISE SAMPLE:
download(1).wav

# NOISE MERGED AUDIO:
download(2).wav

# EXTRACTED NOISE REMOVAL
download(3).wav

# OUTPUT:

![n1](https://github.com/user-attachments/assets/66cf8bf0-8469-4009-8395-62405726059f)
![n2](https://github.com/user-attachments/assets/fb515ee0-d05c-42c6-80af-54537113b9f5)
![n3](https://github.com/user-attachments/assets/6a7a6384-9305-43ba-8b52-6204ab9387be)


# RESULT:
Thus,the analysis of audio signal for noise removal is verified.
