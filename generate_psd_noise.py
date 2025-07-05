import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
from pycbc.noise import noise_from_psd
from pycbc.types import TimeSeries
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def generate_one_noise(args):
    samples_per_segment, delta_t, psd, seed = args
    return noise_from_psd(samples_per_segment, delta_t, psd, seed=seed).numpy()

if __name__ == '__main__':
    # === 1. 打开真实数据文件
    filename = 'H-H1_GWOSC_O3b_4KHZ_R1-1269288960-4096.hdf5'
    with h5py.File(filename, 'r') as f:
        strain = f['strain']['Strain'][()]
        ts = f['strain']['Strain'].attrs['Xspacing']

    sample_rate = 1.0 / ts
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples: {len(strain)}")

    strain_ts = TimeSeries(strain, delta_t=ts)

    # === 2. Welch估计PSD
    segment_duration = 8
    fftlength = 4 # 2
    overlap = 2 # 1
    low_frequency_cutoff = 10.0

    seg_len = int(fftlength * sample_rate)
    seg_stride = int((fftlength - overlap) * sample_rate)

    print("Estimating PSD using Welch method...")
    psd = welch(
        strain_ts,
        seg_len=seg_len,
        seg_stride=seg_stride
    )

    # 计算目标 PSD 点数
    flen = 8193 # 8193
    delta_f = 1.0 / 4  # 4秒段，delta_f = 1/段长

    # 插值 & 平滑
    psd = interpolate(psd, delta_f, length=flen)
    psd = inverse_spectrum_truncation(psd, int(4 * sample_rate), low_frequency_cutoff) # 4 * sample_rate

    # === 3. 裁剪PSD频段
    psd_cleaned = psd.copy()
    psd_cleaned.data[psd_cleaned.sample_frequencies < 10.0] = 0
    psd_cleaned.data[psd_cleaned.sample_frequencies > 1000.0] = 0

    # === 保存裁剪后的PSD和频率到文件
    np.savez('real_psd.npz', psd=psd_cleaned.data, freqs=psd_cleaned.sample_frequencies)
    print("裁剪后的PSD和频率已保存")

    # === 4. 可视化 PSD
    plt.figure(figsize=(8, 5))
    # plt.loglog(psd.sample_frequencies, np.sqrt(psd), label='Original ASD')
    plt.loglog(psd_cleaned.sample_frequencies, np.sqrt(psd_cleaned), linestyle='-')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("ASD (strain/√Hz)")
    plt.title("ASD from real LIGO strain")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === 5. 批量生成噪声（可视化进度）
    duration_each = 10
    n_segments = 4000
    samples_per_segment = int(duration_each * sample_rate)

    print(f"Generating {n_segments} synthetic noise segments of {duration_each}s each (parallel)...")

    args_list = [(samples_per_segment, 1.0 / sample_rate, psd_cleaned, None) for _ in range(n_segments)]

    noise_segments = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_one_noise, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=n_segments):
            noise_segments.append(f.result())

    noise_segments = np.array(noise_segments, dtype=np.float32)
    np.save('colored_noise_val.npy', noise_segments)
    print(f"✅ Saved, shape = {noise_segments.shape}")