import numpy as np
import matplotlib.pyplot as plt
import json

# Parameters
sample_rate = 4096
nperseg = 256
noverlap = 192
n = 1279  # Change this to plot the nth sample
m = 9

# Load signals
gw = np.load('waveforms.npy')
noise = np.load('colored_noise.npy')
injected = np.load('train_X_4s_segments.npy')
processed = np.load('train_X_signals.npy')
X_spectrogram = np.load('train_spectrogram_X.npy')

with open('injection_metadata.json', 'r') as f:
    metadata = json.load(f)

# # 1. Plot gravitational wave signal
# time_axis = np.arange(len(gw[n])) / sample_rate
# plt.figure(figsize=(20, 4))
# plt.plot(time_axis, gw[n])
# plt.title('Gravitational Wave Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()

# # 2. Plot noise signal
# time_axis = np.arange(len(noise[n])) / sample_rate
# plt.figure(figsize=(20, 4))
# plt.plot(time_axis, noise[n])
# plt.title('Noise Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()

# # 3. Plot injected signal
# time_axis = np.arange(len(injected[m])) / sample_rate
# plt.figure(figsize=(20, 4))
# plt.plot(time_axis, injected[m])
# plt.title('Injected Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()

# # 4. Plot processed signal
# time_axis = np.arange(len(processed[m])) / sample_rate
# plt.figure(figsize=(20, 4))
# plt.plot(time_axis, processed[m])
# plt.title('Processed Signal (Whitened + Bandpassed)')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()

# # 4. Plot processed signal around the center
# # 获取信号中心在片段中的时间（秒）
# center_sec = metadata[m]['inject_time_sec'] - metadata[m]['crop_start_sec']
# center_idx = int(center_sec * sample_rate)
# window = int(0.5 * sample_rate)  # 0.1秒对应的采样点数
#
# sig = processed[m]
# start = max(center_idx - window, 0)
# end = min(center_idx + window, len(sig))
#
# time_axis = np.arange(start, end) / sample_rate
# plt.figure(figsize=(16, 4))
# plt.plot(time_axis, sig[start:end])
# plt.title('Processed Signal (Whitened + Bandpassed)')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()

# 5. 绘制完整的时频图
freq_min, freq_max = 20, 300
time_range_sec = 2  # 前后1秒
spec = X_spectrogram[m]
freq_axis = np.linspace(freq_min, freq_max, spec.shape[0])
time_axis = np.linspace(-1, 1, spec.shape[1])

plt.figure(figsize=(8, 4))
plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis',
           extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
plt.colorbar(label='Normalized Power')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f"Spectrogram")
plt.show()
