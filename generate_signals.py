import numpy as np
from pycbc.waveform import get_td_waveform
import json

# ------------------------------
# 参数设置
# ------------------------------
TOTAL_SAMPLES = 2000
SAMPLE_RATE = 4096
DURATION = 4  # seconds
N_POINTS = SAMPLE_RATE * DURATION
F_LOWER = 20.0

# ------------------------------
# 天线模式函数
# ------------------------------
def antenna_patterns(theta, phi, psi):
    """
    Compute detector antenna pattern functions F+ and Fx
    given sky location (theta, phi) and polarization angle psi.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_2psi = np.cos(2 * psi)
    sin_2psi = np.sin(2 * psi)

    Fp = 0.5 * (1 + cos_theta**2) * cos_2psi * np.cos(2 * phi) - cos_theta * sin_2psi * np.sin(2 * phi)
    Fx = 0.5 * (1 + cos_theta**2) * sin_2psi * np.cos(2 * phi) + cos_theta * cos_2psi * np.sin(2 * phi)

    return Fp, Fx

# ------------------------------
# 随机物理参数生成函数
# ------------------------------
def random_parameters():
    m1 = np.random.uniform(10, 80)
    m2 = np.random.uniform(5, m1)
    spin1z = np.random.uniform(-0.99, 0.99)
    spin2z = np.random.uniform(-0.99, 0.99)
    inclination = np.random.uniform(0, np.pi)
    coa_phase = np.random.uniform(0, 2 * np.pi)
    distance = np.random.uniform(100, 1000)  # Mpc
    return {
        'mass1': m1,
        'mass2': m2,
        'spin1z': spin1z,
        'spin2z': spin2z,
        'inclination': inclination,
        'coa_phase': coa_phase,
        'distance': distance
    }

# ------------------------------
# 结果容器
# ------------------------------
waveforms = []
params = []

# ------------------------------
# 主生成循环
# ------------------------------
for i in range(TOTAL_SAMPLES):
    print(f"Generating waveform {i + 1}/{TOTAL_SAMPLES} ...")

    approximant = "SEOBNRv4"

    # 随机物理参数
    p = random_parameters()

    # 随机天空位置和极化角
    theta = np.arccos(np.random.uniform(-1, 1))  # uniform on sphere
    phi = np.random.uniform(0, 2 * np.pi)
    psi = np.random.uniform(0, np.pi)

    # 计算天线模式
    Fp, Fx = antenna_patterns(theta, phi, psi)

    try:
        # 生成波形
        hp, hc = get_td_waveform(
            approximant=approximant,
            mass1=p['mass1'],
            mass2=p['mass2'],
            spin1z=p['spin1z'],
            spin2z=p['spin2z'],
            inclination=p['inclination'],
            coa_phase=p['coa_phase'],
            distance=p['distance'],
            delta_t=1 / SAMPLE_RATE,
            f_lower=F_LOWER
        )

        # 合成为探测器响应
        strain = Fp * hp.numpy() + Fx * hc.numpy()

        # 在4秒窗口里中心对齐
        out_wave = np.zeros(N_POINTS)
        center_idx = N_POINTS // 2
        signal_half = len(strain) // 2
        start = max(center_idx - signal_half, 0)
        end = start + len(strain)
        if end > N_POINTS:
            strain = strain[:N_POINTS - start]
            end = N_POINTS
        out_wave[start:end] = strain

        # 保存
        waveforms.append(out_wave)
        params.append({
            'approximant': approximant,
            **p,
            'sky_theta': theta,
            'sky_phi': phi,
            'polarization_psi': psi,
            'Fp': Fp,
            'Fx': Fx
        })

    except Exception as e:
        print(f"Error generating waveform {i}: {e}")
        continue

# ------------------------------
# 保存结果
# ------------------------------
print("Saving results...")

waveforms = np.array(waveforms)
np.save('waveforms_val.npy', waveforms)

with open('waveform_parameters_val.json', 'w') as f:
    json.dump(params, f, indent=2)

print(f"✅ Saved {len(waveforms)} waveforms with detector projection.")
