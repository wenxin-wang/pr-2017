DATA_DIR=../data/ft1-c-4
MODEL_DIR=$DATA_DIR/model
RECORDS_DIR=$DATA_DIR/records
INCEPTION_CHECKPOINT="../data/inception_v3.ckpt"

python train.py \
  --input_file_pattern="${RECORDS_DIR}/trn-?????-of-00016" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
