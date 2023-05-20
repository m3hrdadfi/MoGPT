#!/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=xxxx-xx-xxxx
#SBATCH --partition=energi
#SBATCH --time=0-12:00:00
#SBATCH --job-name=dd-urb-gpt2-medium-context
#SBATCH --error=logs/dd-urb-context-%J.err.log
#SBATCH --output=logs/dd-urb-context-%J.out.log
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_uni_email

export SEED=101
export USERNAME_DIR=xxxx-xx-xxxx
export ENV_NAME=nlp38
export WANDB_CACHE_DIR=/mimer/NOBACKUP/groups/${USERNAME_DIR}/OUTPUT/.cache/wandb
export TRANSFORMERS_CACHE=/mimer/NOBACKUP/groups/${USERNAME_DIR}/OUTPUT/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/mimer/NOBACKUP/groups/${USERNAME_DIR}/OUTPUT/.cache/huggingface/datasets
export DATASET_MODE=dd

export CUDA_LAUNCH_BLOCKING=1
export PROJECT_NAME=AuxGPT2-Research
export RUN_NAME=${DATASET_MODE}-urb-gpt2-medium-context

export TRAIN_FILE=/mimer/NOBACKUP/groups/${USERNAME_DIR}/DATA/auxgpt-dataset/${DATASET_MODE}/train.json
export VALIDATION_FILE=/mimer/NOBACKUP/groups/${USERNAME_DIR}/DATA/auxgpt-dataset/${DATASET_MODE}/validation.json
export TEST_FILE=/mimer/NOBACKUP/groups/${USERNAME_DIR}/DATA/auxgpt-dataset/${DATASET_MODE}/test.json
export PREDICT_FILE=/mimer/NOBACKUP/groups/${USERNAME_DIR}/DATA/auxgpt-dataset/${DATASET_MODE}/test.json
export IGNORE_INDEX=-100
export LABEL_NAME=labels
export AUX_NAME=aux_labels
export NUM_WORKERS=1
export PREPROCESSING_NUM_WORKERS=1
export BATCH_SIZE=4

#--limit_train_samples=$LIMIT_TRAIN_SAMPLES
export LIMIT_TRAIN_SAMPLES=0
#--limit_val_samples=$LIMIT_VAL_SAMPLES
export LIMIT_VAL_SAMPLES=0
#--limit_test_samples=$LIMIT_TEST_SAMPLES
export LIMIT_TEST_SAMPLES=0
#--limit_predict_samples=$LIMIT_PREDICT_SAMPLES
export LIMIT_PREDICT_SAMPLES=0

export PRETRAINED_MODEL_NAME_OR_PATH=gpt2-medium

#--deepspeed_sharding
export DO_TYPE=BINARY
export DO_ON=CONTEXT
export ALPHA=3.0
# --do_prob_ratio=$DO_PROB_RATIO
export DO_PROB_RATIO=0.15
# --auxiliary_prob_ratio=$REORDER_PROB_RATIO
export REORDER_PROB_RATIO=0.1
export LR=5e-5
export WEIGHT_DECAY=0.01
export NUM_WARMUP_STEPS=5000

export OUTPUT_DIR=/mimer/NOBACKUP/groups/${USERNAME_DIR}/OUTPUT/checkpoints/
export HUB_MODEL_ID=m3hrdadfi/${DATASET_MODE}-urb-gpt2-medium-context
export HUB_TOKEN=xxxx-xx-xxxx
export REPORT_TO=all
export WANDB_TOKEN=xxxx-xx-xxxx
export WANDB_WATCH=all
export WANDB_LOG_MODEL=1

export MAX_EPOCHS=5
export MIX_PRECISION=16
export ACCELERATOR=gpu
export DEVICES=auto
export STRATEGY=None
export LOG_EVERY_N_STEPS=2500
export VAL_CHECK_INTERVAL=2500
export ACCUMULATE_GRAD_BATCHES=1
export GRADIENT_CLIP_VAL=0

module purge
module load git-lfs/3.2.0
module load Anaconda3/2022.05
module load GCC/12.1.0
module load CUDA/11.6.0

git config --global user.name "your_name"
git config --global user.email "your_email"
git config --global credential.helper store

# ENV
echo /mimer/NOBACKUP/groups/${USERNAME_DIR}/envs/${ENV_NAME}
source /apps/Common/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /mimer/NOBACKUP/groups/${USERNAME_DIR}/envs/${ENV_NAME}
wandb login "$WANDB_TOKEN" --relogin

python run_trainer_utterance_reordering.py \
  --output_dir="$OUTPUT_DIR" \
  --project_name="$PROJECT_NAME" \
  --run_name="$RUN_NAME" \
  --train_file="$TRAIN_FILE" \
  --validation_file="$VALIDATION_FILE" \
  --test_file="$TEST_FILE" \
  --predict_file="$PREDICT_FILE" \
  --limit_train_samples=$LIMIT_TRAIN_SAMPLES \
  --limit_val_samples=$LIMIT_VAL_SAMPLES \
  --limit_test_samples=$LIMIT_TEST_SAMPLES \
  --limit_predict_samples=$LIMIT_PREDICT_SAMPLES \
  --ignore_index=$IGNORE_INDEX \
  --label_name="$LABEL_NAME" \
  --aux_name="$AUX_NAME" \
  --batch_size=$BATCH_SIZE \
  --num_workers=$NUM_WORKERS \
  --preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL_NAME_OR_PATH" \
  --alpha=$ALPHA \
  --do_prob_ratio=$DO_PROB_RATIO \
  --reorder_prob_ratio=$REORDER_PROB_RATIO \
  --do_on="$DO_ON" \
  --do_type="$DO_TYPE" \
  --lr=$LR \
  --weight_decay=$WEIGHT_DECAY \
  --num_warmup_steps=$NUM_WARMUP_STEPS \
  --precision="$MIX_PRECISION" \
  --hub_model_id="$HUB_MODEL_ID" \
  --hub_token="$HUB_TOKEN" \
  --report_to="$REPORT_TO" \
  --save_checkpoints \
  --wandb_key="$WANDB_TOKEN" \
  --max_epochs=$MAX_EPOCHS \
  --accelerator="$ACCELERATOR" \
  --devices=$DEVICES \
  --strategy="$STRATEGY" \
  --log_every_n_steps=$LOG_EVERY_N_STEPS \
  --val_check_interval=$VAL_CHECK_INTERVAL \
  --accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
  --gradient_clip_val=$GRADIENT_CLIP_VAL \
  --push_to_hub \
  --tensorboard \
  --with_persona
