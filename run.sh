#!/bin/bash
# ----------------------------------------------------------------------------------------------------
python -u ./ML_code/top.py \
--exp_name $1 \
--train_partition_path /home/PRONTO/datasets/Oracle-NotCompensated-Fine-LLTF/ \
--test_partition_path /home/PRONTO/datasets/Oracle-Compensated-With-Phase-Noise-Large/ \
--save_path /home/nasim/PRONTOFramework/results/ \
--model_flag pronto-l \
--contin true \
--json_path '/home/nasim/PRONTOFramework/results/CFO-Oracle-39k/model_file.json' \
--hdf5_path '/home/nasim/PRONTOFramework/results/CFO-Oracle-39k/model.hdf5' \
--cfo_estimation true \
--cfo_aug true \
--packet_aug false \
--slice_size 160 \
--num_classes 1 \
--batch_size 256 \
--id_gpu $2 \
--normalize true \
--train false \
--test true \
--max_cfo 39000 \
--epochs 2000 \
--early_stopping true \
--patience 1000 \
> /home/nasim/PRONTOFramework/results/$1/log.out \
2> /home/nasim/PRONTOFramework/results/$1/log.err
