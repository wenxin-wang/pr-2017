DATA_DIR=../data/ft1-c-4
MODEL_DIR=$DATA_DIR/model
CHECKPOINT_PATH=$MODEL_DIR/train
RECORDS_DIR=$DATA_DIR/records
VOCAB_FILE=$RECORDS_DIR/word_counts.txt
IMAGE_FILE=../data/image_vgg19_fc1_feature.h5

python run_inference.py \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_file=${IMAGE_FILE} | tee $DATA_DIR/tst-$(date +%F-%H-%M-%S).txt
