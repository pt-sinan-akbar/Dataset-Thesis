# Customer Segmentation using Clustering Algorithms on RFM/RFMD Data

This repository contains the source code for our undergraduate thesis project, where our team compares the performance of various clustering algorithms applied to customer segmentation using RFM (Recency, Frequency, Monetary) and RFMD (Recency, Frequency, Monetary, Demography) data.

## 📌 Project Overview

The goal of this project is to evaluate how different clustering techniques perform in segmenting customers based on their behavior, as represented by RFM or RFMD metrics. This is a common approach in marketing analytics to identify high-value customer groups for targeted campaigns.

We compare several popular clustering algorithms, such as:

- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models (GMM)
- K-Proto

Each algorithm is applied and evaluated on the same dataset using standard metrics (e.g., Silhouette Score, Davies–Bouldin Index, Calinski Harabasz Score).

## 🚀 How to Run
**1. Install Dependencies**
```bash
pip install -r requirements.txt```
```
**2. Set PYTOHNPATH**
- On macOS/Linux
```bash
export PYTHONPATH=$(pwd)
```
- On Windows (Command Line Prompt)
```bash
set PYTHONPATH=%cd%
```
- On Windows (Powershell)
```bash
$env:PYTHONPATH = (Get-Location).Path
```
**3. Run the Program**

Run ```bash python main.py``` to any clustering algorithm you want to run

## 👥 Authors
- Muhammad Sinan Abdul Syakur
- Muhammad Raditya
- Maualana Ahmad Sulami




