import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. 加载模型和数据
model = load_model('gw_cnn_classifier.h5')
X = np.load('GW200129_065458_spectrogram.npy')  # (N, 128, 128) 或 (N, 128, 128, 1)

# 2. 自动扩展通道维
if len(X.shape) == 3:
    X = X[..., np.newaxis]

# 3. 预测概率
probs = model.predict(X, batch_size=32).squeeze()  # (N,)

# 4. 计算对应的时间戳
# 关键参数
gps_start = 1264314069
step_sec = 1  # 原始4秒窗的步长
num_4s_segments = X.shape[0] // 2

window_starts = np.arange(num_4s_segments) * step_sec
time_centers = []
for s in window_starts:
    time_centers.append(s + 1)  # 子段0中心
    time_centers.append(s + 3)  # 子段1中心
time_centers = np.array(time_centers)
gps_times = gps_start + time_centers

# 5. 官方事件GPS
event_gps = 1264316116.4

# 6. 绘制概率流
plt.figure(figsize=(12, 5))
plt.plot(gps_times, probs, marker='o', linestyle='-', label='Predicted Probability')
plt.axvline(event_gps, color='red', linestyle='--', label='Official Event Time')
plt.xlabel('GPS Time (s)')
plt.ylabel('Predicted Probability')
plt.title('GW200129_065458 Spectrogram Probability Flow')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()