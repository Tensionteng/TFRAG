export CUDA_VISIBLE_DEVICES=0

model_name=TFRAG

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/all_datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --fusion_mode 'mean' \
  --w_trend 0 \
  --w_frequency 0 \
  --lradj 'type4'