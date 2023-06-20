CUDA_VISIBLE_DEVICES=2 python ./train.py \
    -m q2p_con \
    --train_query_dir /home/jbai/logical_aser/query_data_filtered/query_data_train_filtered.json \
    --valid_query_dir /home/jbai/logical_aser/query_data_filtered/query_data_valid_filtered.json \
    --test_query_dir /home/jbai/logical_aser/query_data_filtered/query_data_test_filtered.json \
    --checkpoint_path /home/data/jbai/aser_reasoning_logs \
    -b 512 \
    -lr 0.0001 \