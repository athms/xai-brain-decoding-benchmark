#!/usr/bin/env bash
#
#SBATCH -J train            
#SBATCH -o train.output     
#SBATCH -N 1                   
#SBATCH -n 1                   
#SBATCH -p gtx                 
#SBATCH -t 10:00:00            

while [ $# -gt 0 ] ; do
  case $1 in
    --project-dir) PROJ_DIR=$2 ;;
    --task) TASK=$2 ;;
    --data-dir) DATA_DIR=$2 ;;
    --num-hidden-layers) N_HIDDEN=$2 ;;
    --num-filters) N_FILTERS=$2 ;;
    --filter-size) FILTER_SIZE=$2 ;;
    --batch-size) BS=$2 ;;
    --num-epochs) EPOCHS=$2 ;;
    --learning-rate) LR=$2 ;;
    --dropout) DROPOUT=$2 ;;
    --num-runs) RUNS=$2 ;;
    --num-folds) FOLDS=$2 ;;
    --run) RUN=$2 ;;
    --fold) FOLD=$2 ;;
    --stopping-patience) STOP_PATIENCE=$2 ;;
    --stopping-delta) STOP_DELTA=$2 ;;
    --stopping-grace) STOP_GRACE=$2 ;;
    --stopping-plateau-std) STOP_PSTD=$2 ;;
    --stopping-plateau-n) STOP_PN=$2 ;;
    --log-dir) LOG_DIR=$2 ;;
    --run-group-name) RUN_GROUP_NAME=$2 ;;
    --report-to) REPORT_TO=$2 ;;
    --wandb-entity) WANDB_ENTITY=$2 ;;
    --wandb-project) WANDB_PROJECT=$2 ;;
    --wandb-mode) WANDB_MODE=$2 ;;
    --smoke-test) SMOKE_TEST=$2 ;;
    --verbose) VERBOSE=$2 ;;
    --seed) SEED=$2 ;;
    --permute-labels) PERMUTE_LABELS=$2 ;;
    --model-config) MODEL_CONFIG=$2 ;;
    --docker-image-dir) IMAGE_DIR=${2} ;;
  esac
  shift
done

# set defaults
TASK=${TASK:-"WM"}
DATA_DIR=${DATA_DIR:-"${PROJ_DIR}/data/task-${TASK}"}
N_HIDDEN=${N_HIDDEN:-4}
N_FILTERS=${N_FILTERS:-8}
FILTER_SIZE=${FILTER_SIZE:-3}
BS=${BS:-32}
EPOCHS=${EPOCHS:-40}
LR=${LR:-3e-4}
DROPOUT=${DROPOUT:-0.20}
RUNS=${RUNS:-10}
FOLDS=${FOLDS:-3}
RUN=${RUN:--1}
FOLD=${FOLD:--1}
STOP_PATIENCE=${STOP_PATIENCE:-3}
STOP_DELTA=${STOP_DELTA:-0.02}
STOP_GRACE=${STOP_GRACE:-20}
STOP_PSTD=${STOP_PSTD:-0.01}
STOP_PN=${STOP_PN:-10}
LOG_DIR=${LOG_DIR:-"${PROJ_DIR}/results/models/"}
RUN_GROUP_NAME=${RUN_GROUP_NAME:-"none"}
REPORT_TO=${REPORT_TO:-"wandb"}
WANDB_ENTITY=${WANDB_ENTITY:-"athms"}
WANDB_PROJECT=${WANDB_PROJECT:-"interpretability-comparison"}
WANDB_MODE=${WANDB_MODE:-"online"}
SMOKE_TEST=${SMOKE_TEST:-"False"}
VERBOSE=${VERBOSE:-"True"}
SEED=${SEED:-12345}
PERMUTE_LABELS=${PERMUTE_LABELS:-"False"}
MODEL_CONFIG=${MODEL_CONFIG:-"none"}
IMAGE_DIR=${IMAGE_DIR:-"${PROJ_DIR}/images/"}

# TACC-specific imports
module load cuda/11.3
module load tacc-singularity/3.7.2

# set singularity image
IMAGE="${IMAGE_DIR}/interpretability-comparison.simg"
if [[ ! -f $IMAGE ]]; then
    mkdir -p "${IMAGE_DIR}"
    singularity build $IMAGE "docker://arminthomas/interpretability-comparison:dev"
fi

# run
singularity run \
  --nv \
  --cleanenv \
  -B $DATA_DIR:/data:ro \
  $IMAGE \
  python3 scripts/train.py \
    --task $TASK \
    --data-dir /data \
    --num-hidden $N_HIDDEN \
    --num-filters $N_FILTERS \
    --filter-size $FILTER_SIZE \
    --batch-size $BS \
    --num-epochs $EPOCHS \
    --learning-rate $LR \
    --dropout $DROPOUT \
    --num-runs $RUNS \
    --num-folds $FOLDS \
    --run $RUN \
    --fold $FOLD \
    --stopping-patience $STOP_PATIENCE \
    --stopping-delta $STOP_DELTA \
    --stopping-grace $STOP_GRACE \
    --stopping-plateau-std $STOP_PSTD \
    --stopping-plateau-n $STOP_PN \
    --log-dir $LOG_DIR \
    --run-group $RUN_GROUP_NAME \
    --report-to $REPORT_TO \
    --wandb-entity $WANDB_ENTITY \
    --wandb-project $WANDB_PROJECT \
    --wandb-mode $WANDB_MODE \
    --smoke-test $SMOKE_TEST \
    --verbose $VERBOSE \
    --seed $SEED \
    --permute-labels $PERMUTE_LABELS \
    --model-config $MODEL_CONFIG