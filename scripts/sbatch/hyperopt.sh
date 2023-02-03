#!/usr/bin/env zsh

#SBATCH -J hyperopt            # job name
#SBATCH -o hyperopt.output     # output and error file name
#SBATCH -N 1                   # number of nodes requested
#SBATCH -n 1                   # total number of tasks requested
#SBATCH -p gtx                 # partition/queue name (gtx, p100, v100)
#SBATCH -t 05:00:00            # run time (hh:mm:ss)


while [ $# -gt 0 ] ; do
  case $1 in
    --project-dir) PROJ_DIR=$2 ;;
    --task) TASK=${2} ;;
    --data-dir) DATA_DIR=${2} ;;
    --log-dir) LOG_DIR=${2} ;;
    --docker-image-dir) IMAGE_DIR=${2} ;;
  esac
  shift
done

TASK=${TASK:-"WM"}
DATA_DIR=${DATA_DIR:-"${PROJ_DIR}/data/task-${TASK}/trial_images"}
LOG_DIR=${LOG_DIR:-"${PROJ_DIR}/results/hyperopt/task-${TASK}"}

module load cuda/11.0
module load tacc-singularity/3.7.2

IMAGE_DIR=${IMAGE_DIR:-"${PROJ_DIR}/images/"}
IMAGE="${IMAGE_DIR}/interpretability-comparison.simg"
if [[ ! -f $IMAGE ]]; then
    mkdir -p "${IMAGE_DIR}"
    singularity build $IMAGE "docker://arminthomas/interpretability-comparison:rev"
fi

export SINGULARITY_WANDB_USERNAME='athms'
# export SINGULARITY_WANDB_API_KEY=openssl base64 < configs/wandb/wandb_key.txt | tr -d "\n"

mkdir -p $LOG_DIR

singularity run \
  --nv \
  --cleanenv \
  -B $DATA_DIR:/data:ro \
  -B $LOG_DIR:/log_dir \
  $IMAGE \
  python3 scripts/hyperopt.py \
    --task $TASK \
    --data-dir /data \
    --log-dir /log_dir