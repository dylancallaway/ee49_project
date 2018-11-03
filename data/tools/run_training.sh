PIPELINE_CONFIG_PATH=/home/dylan/ee49_project/models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config
TRAIN_DIR=/home/dylan/ee49_project/models/model/train
python object_detection/legacy/train.py \
    --logtostderr --train_dir=${TRAIN_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH}


PIPELINE_CONFIG_PATH=/home/dylan/ee49_project/models/faster_rcnn_nas_coco_2018_01_28/pipeline.config
TRAIN_DIR=/home/dylan/ee49_project/models/model/train
python object_detection/legacy/train.py \
    --logtostderr --train_dir=${TRAIN_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH}



PIPELINE_CONFIG_PATH=/home/dylan/ee49_project/models/faster_rcnn_resnet101_kitti_2018_01_28/pipeline.config
TRAIN_DIR=/home/dylan/ee49_project/models/model/train
python train.py \
    --logtostderr --train_dir=${TRAIN_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH}