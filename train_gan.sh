#!/bin/bash

#Creating kaggle directory
mkdir -p ~/.kaggle

if [ -f $kaggle.json]; then
        echo "kaggle.json found moving to directory"
else
        echo "kaggle.json does not exist, you must create a kaggle.json file in the home directory"
        exit 1
fi

#Installing Kaggle
cp kaggle.json ~/.kaggle/

chmod 600 ~/.kaggle/kaggle.json

pip install kaggle

cd /mydata

kaggle datasets download -d ninadaithal/imagesoasis

unzip /mydata/imagesoasis.zip

#Running python script to prepare for conditional training
python3 dataset_preperation.py

#creating stylegan3 repo
git clone https://github.com/NVlabs/stylegan3.git

mkdir final_dataset
mkdir train_logs

cd stylegan3
#Creating stylegan-ready dataset
python3 dataset_tool.py --source=/mydata/Data/ --dest=/mydata/final_dataset/oasisdataset_256x256.zip --resolution=256x256 --cond=1

#Training stylegan, hyperparameters taken from stylegan github repo
python3 train.py --outdir=/mydata/train_logs --cfg=stylegan2 --data=/mydata/final_dataset/oasisdataset_256x256.zip --gpus=1 --batch=16 --gamma=6.6 --mirror=1 --kimg=3000

echo "Training finished"
