#!/bin/bash
base=/home/kiankd/scratch/hilbert-data/embeddings/
python run_experiments.py dynaglv-hbt-glv-v10k/iter-9400 similarity --base $base --avgvw
#python run_experiments.py STD-W2V-FINAL-v10k similarity --base $base 
#python run_experiments.py STD-GLV-FINAL-v10k similarity --base $base --avgvw
#python run_experiments.py HBT-MLE-FINAL-v10k similarity --base $base
#python run_experiments.py HBT-W2V-FINAL-v10k similarity --base $base
#python run_experiments.py HBT-GLV-FINAL-v10k similarity --base $base --avgvw



