#!/usr/bin/env bash

# Specify the seeds
for seed in {20..40};
do
    sbatch --job-name="groomingToy" ./start.sh $seed
done

