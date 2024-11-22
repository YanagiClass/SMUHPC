#!/bin/sh
#PBS -N job_name
#PBS -V
#PBS -q gpu
#PBS -W block=true
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:01:00
#PBS -o job_stdout.log
#PBS -e job_stderr.log

cd $PBS_O_WORKDIR

./lec12_quiz >& lec12_quiz.log
