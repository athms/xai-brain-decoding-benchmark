#!/usr/bin/env bash
#
#SBATCH -J hyperopt            
#SBATCH -o hyperopt.output     
#SBATCH -N 1                   
#SBATCH -n 1                   
#SBATCH -p gtx                 
#SBATCH -t 05:00:00            

while [ $# -gt 0 ] ; do
  case $1 in
    --project-dir) PROJ_DIR=$2 ;;
    --task) TASK=${2} ;;
    --data-dir) DATA_DIR=${2} ;;
    --hyperopt-dir) HYPEROPT_DIR=${2} ;;
    --docker-image-dir) IMAGE_DIR=${2} ;;
  esac
  shift
done

# set defaults
PROJ_DIR=${PROJ_DIR:-"."}
TASK=${TASK:-"WM"}
DATA_DIR=${DATA_DIR:-"${PROJ_DIR}/data/task-${TASK}"}
HYPEROPT_DIR=${HYPEROPT_DIR:-"${PROJ_DIR}/results/hyperopt/task-${TASK}"}
mkdir -p $HYPEROPT_DIR
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
  -B $HYPEROPT_DIR:/hyperopt_dir \
  $IMAGE \
  python3 scripts/hyperopt.py \
    --task $TASK \
    --data-dir /data \
    --hyperopt-dir /hyperopt_dir