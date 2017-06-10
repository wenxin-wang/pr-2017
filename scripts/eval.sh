DATA_DIR=../data/ft1-c-2
MODEL_DIR=$DATA_DIR/model
RECORDS_DIR=$DATA_DIR/records

python evaluate.py \
  --input_file_pattern="${RECORDS_DIR}/val-?????-of-00002" \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --eval_dir="${MODEL_DIR}/eval"
