PIPELINE_CONFIG_PATH=models/faster_rcnn_resnet50_lowproposals_coco_2018_01_28/pipeline.config
TRAIN_DIR=models/faster_rcnn_resnet50_lowproposals_coco_2018_01_28/train
python train.py \
    --logtostderr --train_dir=${TRAIN_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH}