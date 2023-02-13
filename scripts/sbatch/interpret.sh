#!/usr/bin/env bash
#
#SBATCH -J interpret            
#SBATCH -o interpret.output     
#SBATCH -N 1                   
#SBATCH -n 1                   
#SBATCH -p gtx                 
#SBATCH -t 10:00:00            

while [ $# -gt 0 ] ; do
  case $1 in
    --project-dir) PROJ_DIR=$2 ;;
    --task) TASK=${2} ;;
    --fitted-model-dir) MODEL_DIR=${2} ;;
    --data-dir) DATA_DIR=${2} ;;
    --attributions-dir) ATTR_DIR=${2} ;;
    --use-random-init) RAND_INIT=${2} ;;
    --interpret-final-model FINAL_MODEL=${2} ;;
    --docker-image-dir) IMAGE_DIR=${2} ;;
  esac
  shift
done

# set defaults
PROJ_DIR=${PROJ_DIR:-"."}
TASK=${TASK:-"WM"}
MODEL_DIR=${MODEL_DIR:-"${PROJ_DIR}/results/models/task-${TASK}"}
DATA_DIR=${DATA_DIR:-"${PROJ_DIR}/data/task-${TASK}"}
ATTR_DIR=${ATTR_DIR:-"${PROJ_DIR}/results/attributions/task-${TASK}"}
mkdir -p $ATTR_DIR
RAND_INIT=${RAND_INIT:-"False"}
FINAL_MODEL=${FINAL_MODEL:-"False"}
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
  -B $MODEL_DIR:/model:ro \
  -B $DATA_DIR:/data:ro \
  -B $ATTR_DIR:/attr \
  $IMAGE \
  python3 scripts/interpret.py \
    --task $TASK \
    --fitted-model-dir /model \
    --data-dir /data \
    --attributions-dir /attr \
    --use-random-init $RAND_INIT \
    --interpret-final-model $FINAL_MODEL