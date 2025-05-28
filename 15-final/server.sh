#!/bin/bash

source /home/ubuntu/skripsi-venv/bin/activate
export PYTHONPATH=/home/ubuntu/Dataset-Thesis

echo "Welcome to the jungle"
date

echo "K-Means"
date
python KMEANS.py > KMEANS_output.txt 2>&1
echo "K-Proto"
date
python KPROTO.py > KPROTO_output.txt 2>&1
echo "DBSCAN"
date
python DBSCAN.py > DBSCAN_output.txt 2>&1
echo "GMM"
date
python GMM.py > GMM_output.txt 2>&1
echo "Hierarchical"
python HIERARCHICAL.py > HIERARCHICAL_output.txt 2>&1

date
echo "Dan kita telah resmi menamatkan..."

