#!/bin/bash
# Single usage: python3 seg.py src.jpg dst.png

# for f in ./dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data/{train,test,validation}/*/ ; do
#     folder="${f:0:77}_removebg${f:77}"    
#     mkdir -p "$folder"
# done

for f in ./dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data/train/*/*; do
    dest="${f:0:77}_removebg${f:77:-3}png"
    
    if [[ ! -f "$dest" ]]; then
        python3 seg.py "$f" "$dest"
    fi

done
