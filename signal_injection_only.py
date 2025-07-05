import numpy as np
import json
from tqdm import tqdm

# === 参数 ===
sample_rate = 4096
noise_duration = 10
crop_duration = 4
n_total_segments = 4000
n_injections_target = 2000
n_noise_target = 2000

n_samples_noise = noise_duration * sample_rate
n_samples_crop = crop_duration * sample_rate
injection_range = (2.0, 6.0)

# === 1. 加载数据 ===
print("Loading data...")
noise_data = np.load('colored_noise_val.npy')
gw_signals = np.load('waveforms_val.npy')
with open('waveform_parameters_val.json', 'r') as f:
    waveform_params = json.load(f)

# === 2. 预分配输出 ===
final_X = []
final_y = []
injection_metadata = []

all_noise_indices = np.arange(len(noise_data))
all_signal_indices = np.arange(len(gw_signals))

np.random.shuffle(all_noise_indices)

print("Building dataset (no SNR filtering)...")

n_injections_done = 0
n_noise_done = 0
attempts = 0
max_attempts = n_total_segments * 10

progress = tqdm(total=n_total_segments, desc="Collecting samples")

while (n_injections_done < n_injections_target or n_noise_done < n_noise_target) and attempts < max_attempts:
    attempts += 1

    if n_injections_done < n_injections_target and (np.random.rand() < 0.5 or n_noise_done >= n_noise_target):
        # Try injection
        noise_idx = np.random.choice(all_noise_indices)
        noise = noise_data[noise_idx].copy()

        signal_idx = np.random.choice(all_signal_indices)
        signal = gw_signals[signal_idx]
        signal_params = waveform_params[signal_idx]
        signal_len = len(signal)

        t_inject = np.random.uniform(*injection_range)
        inject_sample = int(t_inject * sample_rate)
        start = inject_sample - signal_len // 2
        end = start + signal_len

        # 防止越界
        if start < 0:
            shift = -start
            start = 0
            end += shift
            signal = signal[shift:]
        if end > n_samples_noise:
            cut = end - n_samples_noise
            end = n_samples_noise
            signal = signal[:-cut]

        noise[start:end] += signal

        # 裁剪4秒窗口
        min_crop_start = max(0, inject_sample - n_samples_crop + signal_len // 2)
        max_crop_start = min(n_samples_noise - n_samples_crop, inject_sample - signal_len // 2)
        if max_crop_start <= min_crop_start:
            crop_start = min_crop_start
        else:
            crop_start = np.random.randint(min_crop_start, max_crop_start)
        crop_end = crop_start + n_samples_crop
        cropped = noise[crop_start:crop_end]

        final_X.append(cropped.astype(np.float32))
        final_y.append(1)
        injection_metadata.append({
            "type": "injection",
            "original_noise_idx": int(noise_idx),
            "inject_waveform_idx": int(signal_idx),
            "inject_time_sec": float(t_inject),
            "crop_start_sec": float(crop_start / sample_rate),
            "params": signal_params
        })
        n_injections_done += 1
        progress.update(1)
    else:
        # Pure noise
        noise_idx = np.random.choice(all_noise_indices)
        noise = noise_data[noise_idx]
        max_start = n_samples_noise - n_samples_crop
        crop_start = np.random.randint(0, max_start)
        crop_end = crop_start + n_samples_crop
        cropped = noise[crop_start:crop_end]

        final_X.append(cropped.astype(np.float32))
        final_y.append(0)
        injection_metadata.append({
            "type": "noise",
            "original_noise_idx": int(noise_idx),
            "inject_waveform_idx": None,
            "inject_time_sec": None,
            "crop_start_sec": float(crop_start / sample_rate),
            "params": None
        })
        n_noise_done += 1
        progress.update(1)

progress.close()

print(f"✅ Collected {n_injections_done} injections and {n_noise_done} noise segments.")

# === 3. 保存输出 ===
final_X = np.stack(final_X)
final_y = np.array(final_y, dtype=np.int32)

np.save('validation_X_4s_segments.npy', final_X)
np.save('validation_y_labels.npy', final_y)
with open('injection_metadata_val.json', 'w') as f:
    json.dump(injection_metadata, f, indent=2)

print("✅ Done! All data saved.")
