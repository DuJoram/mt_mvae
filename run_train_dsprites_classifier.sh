#!/usr/bin/env bash
#BSUB -n 16
#BSUB -W 16:00
#BSUB -R "rusage[mem=1024,scratch=1024,ngpus_excl_p=1,]"
#BSUB -J "joraml_train_dsprites_classifier"
#BSUB -o train_dsprites_classifier_%J.out


JOB_NAME="train_dsprites_classifier_${LSB_JOBID}"
OUTPUT_DIR=${TMPDIR}/runs/${JOB_NAME}
RUN_OUTPUT_TARGET=${LS_SUBCWD}/runs/${JOB_NAME}.tar.gz

LEARNING_RATE=1e-3
BATCH_SIZE=4096
EPOCHS=500
EVALUATION_FREQUENCY=10
CHECKPOINT_FREQUNECY=10
OUTPUT_DIR=${OUTPUT_DIR}
NUM_WORKERS=16

echo "TMPDIR = $TMPDIR"
echo "LS_SUBCWD = $LS_SUBCWD"
echo "LSB_JOBID = $LSB_JOBID"
echo "JOB_NAME = $JOB_NAME"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "RUN_OUTPUT_TARGET = $RUN_OUTPUT_TARGET"

mkdir -p $TMPDIR/resources/data

cd $TMPDIR || exit 1

rsync -aq ${LS_SUBCWD}/src ./
rsync -aq ${LS_SUBCWD}/resources/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz ./resources/data/


eval "$(conda shell.bash hook)"
conda activate mt_mvae
echo "CONDA_PREFIX: ${CONDA_PREFIX}"


python src/train_dsprites_classifier.py --output-dir=$OUTPUT_DIR \
  --dsprites-archive-path "resources/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"\
  --learning-rate $LEARNING_RATE \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --evaluation-frequency $EVALUATION_FREQUENCY \
  --checkpoint-frequency $CHECKPOINT_FREQUNECY \
  --output-dir $OUTPUT_DIR \
  --num-workers $NUM_WORKERS


function cleanup {
  tar -czf "${RUN_OUTPUT_TARGET}" -C "${TMPDIR}/runs/" "${JOB_NAME}"
}

trap cleanup EXIT

trap "echo 'Received USR2; ignoring'" USR2
trap "echo 'Received INT; ignoring'" INT
trap "echo 'Received QUIT; ignoring'" QUIT
trap "echo 'Received TERM; ignoring'" TERM
