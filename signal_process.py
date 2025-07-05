import numpy as np
from scipy.signal import butter, filtfilt
from pycbc.types import FrequencySeries

# 参数
sample_rate = 4096
lowcut = 10.0 # 20.0 时域好看波形
highcut = 1000.0 # 300.0 时域好看波形

# 读取信号和PSD
gw_signals = np.load('GW200129_065458_segments_4s.npy')
psd_npz = np.load('real_psd.npz')
psd_data = psd_npz['psd']
psd_freqs = psd_npz['freqs']
# 将 PSD 中的 0 替换为一个很小的正数
psd_data[psd_data == 0] = 1e-20

n_samples = gw_signals.shape[1]
flen = n_samples // 2 + 1
delta_f = 1.0 / (n_samples / sample_rate)
psd = FrequencySeries(psd_data[:flen], delta_f=delta_f)

def whiten_manual(signal, psd, sample_rate, epsilon=0):
    n = len(signal)
    # 傅里叶变换
    freqs = np.fft.rfftfreq(n, 1/sample_rate)
    sig_fft = np.fft.rfft(signal)
    # PSD 可能比信号短，做截断或插值
    if len(psd) != len(sig_fft):
        psd_interp = np.interp(freqs, np.linspace(0, sample_rate/2, len(psd)), psd)
    else:
        psd_interp = psd
    # 白化
    white_fft = sig_fft / (np.sqrt(psd_interp) + epsilon)
    # 逆变换
    white = np.fft.irfft(white_fft, n)
    return white

def bandpass(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

whitened_bandpassed = []
for sig in gw_signals:
    w = whiten_manual(sig, psd_data[:flen], sample_rate)
    bp = bandpass(w, lowcut, highcut, sample_rate)
    whitened_bandpassed.append(bp.astype(np.float32))

whitened_bandpassed = np.stack(whitened_bandpassed)
np.save('GW200129_065458_processed.npy', whitened_bandpassed)