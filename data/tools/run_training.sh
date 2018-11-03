PIPELINE_CONFIG_PATH=/home/dylan/ee49_project/models/faster_rcnn_resnet101_kitti_2018_01_28/pipeline.config
TRAIN_DIR=/home/dylan/ee49_project/models/model/train
python train.py \
    --logtostderr --train_dir=${TRAIN_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH}