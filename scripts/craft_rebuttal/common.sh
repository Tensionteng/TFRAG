#!/bin/bash
# Shared config for the CRAFT rebuttal campaign. Sourced by every b*.sh script.
#
# Override anything from the environment, e.g.
#   SEEDS="1 2 3" GPU=1 bash scripts/craft_rebuttal/b2_main_multiseed.sh

set -u

# ---------------------------------------------------------------- environment
PY=${PY:-"uv run python"}          # set PY="python" if you manage the env yourself
GPU=${GPU:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
SEEDS=${SEEDS:-"2021 1 2 3 4"}     # 5 seeds; 2021 first so it reproduces the old default
PRED_LENS=${PRED_LENS:-"96 192 336 720"}
EPOCHS=${EPOCHS:-10}               # the ONE training budget for the whole campaign
PATIENCE=${PATIENCE:-3}
DES=${DES:-Rebuttal}
LOGDIR=${LOGDIR:-logs}
DRY_RUN=${DRY_RUN:-0}
# Disk hygiene. pred/true arrays cost ~160 MB per Weather run and are only needed by
# the frequency analysis; the wrapper checkpoint is redundant once base_model.pth is
# written. SLIM=1 drops both -- necessary on a nearly full disk.
SLIM=${SLIM:-0}
ARTIFACT_FLAGS=""
[ "$SLIM" = "1" ] && ARTIFACT_FLAGS="--no_save_arrays --slim_ckpt"

mkdir -p "$LOGDIR" runs analysis

# ------------------------------------------------------------------- datasets
# name|root_path|data_path|data|enc_in
DATASETS_ALL=(
  "ETTh1|./dataset/ETT-small/|ETTh1.csv|ETTh1|7"
  "ETTh2|./dataset/ETT-small/|ETTh2.csv|ETTh2|7"
  "ETTm1|./dataset/ETT-small/|ETTm1.csv|ETTm1|7"
  "ETTm2|./dataset/ETT-small/|ETTm2.csv|ETTm2|7"
  "Weather|./dataset/weather/|weather.csv|custom|21"
  "Exchange|./dataset/exchange_rate/|exchange_rate.csv|custom|8"
  "ECL|./dataset/electricity/|electricity.csv|custom|321"
  "Traffic|./dataset/traffic/|traffic.csv|custom|862"
)
# Small/fast subset, for pilots.
DATASETS_SMALL=(
  "ETTh1|./dataset/ETT-small/|ETTh1.csv|ETTh1|7"
  "Weather|./dataset/weather/|weather.csv|custom|21"
)

# --------------------------------------------------------- per-model defaults
# Kept identical between base and CRAFT runs: the comparison must differ only in
# the corrector. lr/batch/d_model come from the TSLib reference configs.
model_args() {
  local model=$1 dataset=$2
  case "$model" in
    iTransformer)
      # Reference configs from this repo's own scripts/long_term_forecast/*/iTransformer.sh.
      # Traffic uses e_layers 4 there, not 3.
      if [ "$dataset" = "Traffic" ]; then
        echo "--e_layers 4 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 --n_heads 8"
      else
        echo "--e_layers 3 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 --n_heads 8"
      fi
      ;;
    PatchTST)     echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 128 --d_ff 128 --n_heads 8" ;;
    DLinear)      echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 128 --d_ff 128 --n_heads 8" ;;
    TimeXer)      echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 --n_heads 8 --patch_len 16" ;;
    WPMixer)      echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 256 --d_ff 256 --n_heads 8" ;;
    MultiPatchFormer) echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 256 --d_ff 256 --n_heads 8" ;;
    TimeMixer)    echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 128 --d_ff 128 --n_heads 8 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2" ;;
    TimesNet)     echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 64 --d_ff 64 --n_heads 8 --top_k 5" ;;
    # ICLR 2026 baselines, configs from the authors' own scripts (FACT/FACT.sh,
    # MixLinear/scripts/MixLinear/*.sh). Added for reviewer ErQJ's "outdated baselines".
    FACT)
      case "$dataset" in
        Traffic) echo "--e_layers 1 --d_model 1024 --d_ff 1024 --core 0.5 --dilation 1 2 3 2 1 --num_kernels 4 --use_norm 1" ;;
        ECL)     echo "--e_layers 1 --d_model 512 --d_ff 2048 --core 0.1 --dilation 1 2 3 4 3 2 1 --num_kernels 4 --use_norm 1" ;;
        *)       echo "--e_layers 1 --d_model 512 --d_ff 512 --core 0.5 --dilation 1 2 1 --num_kernels 4 --use_norm 1" ;;
      esac ;;
    MixLinear)
      # period_len/lpf/alpha per the authors' *_best.sh; d_model unused by the model.
      case "$dataset" in
        ETTh2)   echo "--period_len 24 --lpf 19 --alpha 0.5 --e_layers 1 --d_model 128 --d_ff 128" ;;
        ETTm1)   echo "--period_len 2 --lpf 15 --alpha 0.01 --e_layers 1 --d_model 128 --d_ff 128" ;;
        Weather) echo "--period_len 4 --lpf 15 --alpha 0.01 --e_layers 1 --d_model 128 --d_ff 128" ;;
        ECL|Traffic) echo "--period_len 24 --lpf 19 --alpha 0.5 --e_layers 1 --d_model 128 --d_ff 128" ;;
        *)       echo "--period_len 24 --lpf 1 --alpha 0.95 --e_layers 1 --d_model 128 --d_ff 128" ;;
      esac ;;
    *)            echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 128 --d_ff 128 --n_heads 8" ;;
  esac
}

# Training budget. Each model gets the budget its own reference script uses, applied
# identically to that model's base and CRAFT arms so every paired comparison stays
# matched. Cross-model rows differ in budget exactly as the source papers do.
epochs_for() {
  case "$1" in
    FACT)      echo "15" ;;
    MixLinear) echo "30" ;;
    *)         echo "$EPOCHS" ;;
  esac
}

# Learning rates taken from this repo's reference scripts, NOT guessed. Where a
# reference script sets no --learning_rate it inherits run.py's default of 1e-4, which
# is the case for Weather, Exchange and the ETT family.
#
# This was wrong in an earlier version of this file: Weather ran at 5e-4 (5x the
# reference) and Traffic at 5e-4 (half the reference 1e-3). A too-high LR destabilises
# the MSE baseline, which then flatters any objective with bounded gradients -- MAE and
# Huber appeared to beat MSE by ~20% on Weather purely for that reason. Do not change
# these without checking the reference script.
lr_for() {  # lr_for <dataset> [model]
  local ds=$1 model=${2:-iTransformer}
  case "$model" in
    FACT)      echo "0.001"; return ;;                       # FACT.sh
    MixLinear)                                               # MixLinear/scripts/*
      case "$ds" in
        ETTm1)         echo "0.005" ;;
        ETTh2|Weather) echo "0.02"  ;;
        *)             echo "0.03"  ;;
      esac; return ;;
  esac
  case "$ds" in
    ECL)      echo "0.0005" ;;   # ECL_script/iTransformer.sh
    Traffic)  echo "0.001"  ;;   # Traffic_script/iTransformer.sh
    *)        echo "0.0001" ;;   # run.py default; Weather/Exchange/ETT reference scripts set none
  esac
}

bs_for() {  # bs_for <dataset> [model]
  local ds=$1 model=${2:-iTransformer}
  case "$model" in
    FACT) echo "32"; return ;;
    MixLinear)
      case "$ds" in
        ECL|Traffic) echo "64"  ;;
        *)           echo "256" ;;
      esac; return ;;
  esac
  case "$ds" in
    Traffic|ECL) echo "16" ;;
    *)           echo "32" ;;
  esac
}

# Large-variate datasets keep the retrieved-future store in host RAM.
mem_flags_for() {
  case "$1" in
    ECL|Traffic) echo "--memory_store_cpu" ;;
    *)           echo "" ;;
  esac
}

# ------------------------------------------------------------------- launcher
# run_one <name> <root> <data_path> <data> <enc_in> <model> <pred_len> <seed> [extra...]
run_one() {
  local name=$1 root=$2 dpath=$3 data=$4 enc=$5 model=$6 pl=$7 seed=$8
  shift 8
  local extra="$*"
  local lr bs ep sl
  lr=$(lr_for "$name" "$model")
  bs=$(bs_for "$name" "$model")
  ep=$(epochs_for "$model")
  sl=${SEQ_LEN:-96}
  local tagsafe
  tagsafe=$(echo "${name}_${model}_pl${pl}_s${seed}_$(echo "$extra" | tr -cd '[:alnum:]')" | cut -c1-120)
  local log="$LOGDIR/${tagsafe}.log"

  local cmd="$PY -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path $root --data_path $dpath --data $data \
    --model_id ${name}_96_${pl} --model $model --features M \
    --seq_len $sl --label_len 48 --pred_len $pl \
    --enc_in $enc --dec_in $enc --c_out $enc \
    $(model_args "$model" "$name") \
    --learning_rate $lr --batch_size $bs \
    --train_epochs $ep --patience $PATIENCE --lradj type1 \
    --seed $seed --des $DES --itr 1 \
    --num_workers $NUM_WORKERS --gpu $GPU \
    $ARTIFACT_FLAGS $(mem_flags_for "$name") $extra"

  if [ "$DRY_RUN" = "1" ]; then
    echo "$cmd"
    return 0
  fi
  echo "=== $tagsafe"
  if eval "$cmd" > "$log" 2>&1; then
    tail -2 "$log" | sed 's/^/    /'
  else
    echo "    !! FAILED -- see $log"
    tail -5 "$log" | sed 's/^/    /'
  fi
}

# CRAFT flags. Exclusion radius = pred_len, so the query's target window can never
# overlap a retrieved neighbour's target window. This is the paper's safeguard,
# which the released code did not implement.
craft_flags() {
  local pl=$1
  echo "--use_rag --gamma_1 1.0 --gamma_2 ${GAMMA2:-0.5} \
        --num_retrieve ${K:-5} --num_rl_samples ${NS:-8} \
        --exclusion_radius $pl --retrieval_mode nn --reward_type discrete"
}
