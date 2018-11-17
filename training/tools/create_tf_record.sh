python create_tf_record.py \
    --data_dir=/home/dylan/ee49_project/training/data/raw/hands \
    --set=train \
    --output_path=/home/dylan/ee49_project/training/data/tf_records/hands/hands_train.record \
    --label_map_path=/home/dylan/ee49_project/training/data/tf_records/hands/label_map.pbtxt
