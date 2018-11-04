PIPELINE_CONFIG_PATH=/home/dylan/ee49_project/models/model/pipeline.config
TRAIN_DIR=/home/dylan/ee49_project/models/model/train
python train.py \
    --logtostderr --train_dir=${TRAIN_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH}