import numpy as np
import json
from scipy.signal import spectrogram
from tqdm import tqdm
from skimage.transform import resize

# === 参数配置 ===
input_file_X = 'validation_X_signals.npy'
input_file_y = 'validation_y_labels.npy'
metadata_file = 'injection_metadata_val.json'
output_file_X = 'validation_spectrogram_X.npy'
output_file_y = 'validation_spectrogram_y.npy'

sample_rate = 4096
nperseg = 256
noverlap = 192
target_shape = (128, 128) # 目标分辨率
freq_min, freq_max = 20, 300 # 频率范围（Hz）
window_sec = 1 # 注入点前后窗口长度（秒）
# gamma = 0.6 # 伽玛校正参数

# === 1. 加载数据 ===
X_time = np.load(input_file_X)
y_labels = np.load(input_file_y)
n_samples = X_time.shape[0]

# 读取注入元数据
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# === 2. 预探测频率轴 ===
f_probe, t_probe, Sxx = spectrogram(X_time[0], fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
freq_mask = (f_probe >= freq_min) & (f_probe <= freq_max)
freq_bins = freq_mask.sum()

# === 3. 预分配输出 ===
X_spectrogram = np.zeros((n_samples, *target_shape), dtype=np.float32)

# === 4. 生成时频图 ===
for i in tqdm(range(n_samples)):
    f, t, Sxx = spectrogram(
        X_time[i],
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap
    )
    log_Sxx = np.log1p(Sxx)
    cropped_freq = log_Sxx[freq_mask, :]

    # 判断是否有注入
    meta = metadata[i] if i < len(metadata) else None
    if meta and meta.get("type") == "injection":
        inject_time = meta['inject_time_sec'] - meta['crop_start_sec']
    else:
        inject_time = np.random.uniform(1.0, 3.0)  # 纯噪声在 [1, 3] 秒内随机

    # 时间裁剪
    time_mask = (t >= inject_time - window_sec) & (t <= inject_time + window_sec)
    cropped = cropped_freq[:, time_mask]

    # 伽玛校正
    # cropped = cropped ** gamma
    # 尺寸重采样
    resized = resize(cropped, target_shape, mode='reflect', anti_aliasing=True)
    X_spectrogram[i] = resized

# === 5. 全局归一化 ===
global_min = X_spectrogram.min()
global_max = X_spectrogram.max()
print(f"Global min: {global_min}, Global max: {global_max}")
X_spectrogram = (X_spectrogram - global_min) / (global_max - global_min)

# === 6. 保存 ===
np.save(output_file_X, X_spectrogram)
np.save(output_file_y, y_labels)
print(f"✅ Saved spectrogram dataset to {output_file_X}, labels to {output_file_y}.")