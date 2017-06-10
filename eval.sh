DATA_DIR=../data/ft1-c-4
MODEL_DIR=$DATA_DIR/model
RECORDS_DIR=$DATA_DIR/records

python evaluate.py \
  --input_file_pattern="${RECORDS_DIR}/trn-?????-of-00016" \
  --train_dir="${MODEL_DIR}/train" \
  --number_of_steps=1000000
