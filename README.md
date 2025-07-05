# GW-detection-by-ML

This repository contains code for an end-to-end deep learning pipeline that detects gravitational-wave (GW) signals in LIGO strain data using 2D spectrogram images and a convolutional neural network (CNN) classifier.

It demonstrates a lightweight CNN trained on simulated BBH injections into realistic noise, with PSD whitening and time-frequency preprocessing. The trained model is also tested on real LIGO data to identify known events via a sliding window scan.

## 📜 Features

- Generates 2D spectrogram images from whitened strain data
- Lightweight 2D CNN model with early stopping and learning rate scheduler
- Physically informed sample weighting based on chirp mass and distance
- Sliding-window search on real LIGO strain data with GPS timestamp alignment
- Visualization of detection probability time series with known-event markers

---
