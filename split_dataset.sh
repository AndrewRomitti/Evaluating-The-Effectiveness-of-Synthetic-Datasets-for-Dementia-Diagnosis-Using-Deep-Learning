#!/bin/bash

cd /mydata
cp -r /mydata/Data /mydata/Data_Copy

echo "What dataset split would you like to create? Please type: 25%"
read $decision

if[$decision=="25%"]; then

        #Creating synthetic directory that mimicks real directories
        mkdir Synthetic_25
        cd Synthetic_25
        mkdir Mild_Dementia
        mkdir Moderate_Dementia
        mkdir Non_Demented
        mkdir Very_mild_Dementia

        cd --
        cd /mydata

        #generating images per each class, images mimick 25% of real dataset composition per class
        cd stylegan3
        python3 gen_images.py --outdir=/mydata/Synthetic_25/Mild_Dementia --trunc=1 --seeds=0-1250 --network=/mydata/train_logs/training-runs/05000-stylegan2-imagesoasis256x256-gpus1-batch16-gamma6.6/network-snapshot-5000.pkl --class=0
        python3 gen_images.py --outdir=/mydata/Synthetic_25/Moderate_Dementia --trunc=1 --seeds=0-122 --network=/mydata/train_logs/training-runs/05000-stylegan2-imagesoasis256x256-gpus1-batch16-gamma6.6/network-snapshot-5000.pkl --class=1
        python3 gen_images.py --outdir=/mydata/Synthetic_25/Non_Demented --trunc=1 --seeds=0-16800 --network=/mydata/train_logs/training-runs/05000-stylegan2-imagesoasis256x256-gpus1-batch16-gamma6.6/network-snapshot-5000.pkl --class=2
        python3 gen_images.py --outdir=/mydata/Synthetic_25/Very_mild_Dementia --trunc=1 --seeds=0-3425 --network=/mydata/train_logs/training-runs/05000-stylegan2-imagesoasis256x256-gpus1-batch16-gamma6.6/network-snapshot-5000.pkl --class=3

