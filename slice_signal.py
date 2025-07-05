import numpy as np
import h5py

sample_rate = 4096  # 采样率
segment_length_sec = 4  # 每段长度（秒）
step_sec = 1  # 滑窗步长（秒）

# === 参数换算 ===
segment_length = segment_length_sec * sample_rate  # 每段采样点数
step = step_sec * sample_rate  # 步长采样点数

# === 读取数据 ===
with h5py.File('GW200129_065458.hdf5', 'r') as hf:
    strain = np.array(hf['strain/Strain'])

# === 滑窗分段 ===
segments = []
for start in range(0, len(strain) - segment_length + 1, step):
    segment = strain[start : start + segment_length]
    segments.append(segment)

segments = np.array(segments)
print(f"Total segments: {len(segments)}, shape: {segments.shape}")

# === 保存 ===
np.save('GW200129_065458_segments_4s.npy', segments)