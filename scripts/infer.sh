# DATA_DIR=../data/ft1-w-4
# MODEL_DIR=$DATA_DIR/model
# #MODEL_DIR=$DATA_DIR/model/orig-06-10-14-37
# CHECKPOINT_PATH=$MODEL_DIR/train
# RECORDS_DIR=$DATA_DIR/records
# VOCAB_FILE=$RECORDS_DIR/word_counts.txt
# IMAGE_FILE=../data/image_vgg19_fc1_feature.h5
# 
# #export CUDA_VISIBLE_DEVICES=""
# 
# python run_inference.py \
#   --checkpoint_path=${CHECKPOINT_PATH} \
#   --vocab_file=${VOCAB_FILE} \
#   --input_file=${IMAGE_FILE} | tee $DATA_DIR/tst-$(date +%F-%H-%M-%S).txt

DATA_DIR=../data/cnn-c-3
MODEL_DIR=$DATA_DIR/model
#MODEL_DIR=$DATA_DIR/model/orig-06-10-14-37
CHECKPOINT_PATH=$MODEL_DIR/train
RECORDS_DIR=$DATA_DIR/records
VOCAB_FILE=$RECORDS_DIR/word_counts.txt
IMAGE_FILE=../data/image_vgg19_fc1_feature.h5

#export CUDA_VISIBLE_DEVICES=""

python run_inference.py \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_file=${IMAGE_FILE} | tee $DATA_DIR/tst-$(date +%F-%H-%M-%S).txt
