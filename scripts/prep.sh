python preprocess.py --output_dir=../data/cnn-c-3/records/ \
	--ft1=0 --ft2=0 --cnn=1 \
	--num_threads=8 \
	--min_word_count=3 \
	--train_shards=128 \
	--val_shards=16
