import numpy as np
from scipy.signal import spectrogram
from skimage.transform import resize
from tqdm import tqdm

input_file = 'GW200129_065458_processed.npy'
output_file = 'GW200129_065458_spectrogram.npy'

sample_rate = 4096
nperseg = 256
noverlap = 192
target_shape = (128, 128)
freq_min, freq_max = 20, 300
# gamma = 0.6

# 加载数据
X_time = np.load(input_file)  # (num_segments, segment_length)
num_segments, segment_length = X_time.shape
half_len = segment_length // 2  # 2秒

# 预探测频率轴
f_probe, t_probe, Sxx = spectrogram(X_time[0, :half_len], fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
freq_mask = (f_probe >= freq_min) & (f_probe <= freq_max)

# 预分配输出
X_spectrogram = np.zeros((num_segments * 2, *target_shape), dtype=np.float32)

for i in tqdm(range(num_segments)):
    for j, part in enumerate([X_time[i, :half_len], X_time[i, half_len:]]):
        f, t, Sxx = spectrogram(part, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        log_Sxx = np.log1p(Sxx)
        cropped_freq = log_Sxx[freq_mask, :]
        # 伽玛校正
        cropped = cropped_freq # ** gamma
        # 尺寸重采样
        resized = resize(cropped, target_shape, mode='reflect', anti_aliasing=True)
        X_spectrogram[i * 2 + j] = resized

# 全局归一化
global_min = X_spectrogram.min()
global_max = X_spectrogram.max()
print(f"Global min: {global_min}, Global max: {global_max}")
X_spectrogram = (X_spectrogram - global_min) / (global_max - global_min)

np.save(output_file, X_spectrogram)
print(f"✅ Saved spectrograms to {output_file}, shape: {X_spectrogram.shape}")