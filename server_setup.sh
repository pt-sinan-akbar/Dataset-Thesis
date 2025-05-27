#!/bin/bash

sudo apt update 
sudo apt install -y python3-pip python3.12-venv neofetch speedtest-cli btop
sudo timedatectl set-timezone Asia/Jakarta
python3 -m venv /home/ubuntu/skripsi-venv
source /home/ubuntu/skripsi-venv/bin/activate
pip install -r /home/ubuntu/Dataset-Thesis/requirements.txt

