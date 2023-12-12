#!/bin/bash
#SBATCH -t 0-t:00
#SBATCH -p partition
#SBATCH --gres=gpu:1
#SBATCH --mem=memory
module load modules
source activate environment


mkdir data
cd data
git clone https://github.com/jayaneetha/colorized-MNIST
mkdir hard_mnist
cd hard_mnist
mkdir training
cd training
mkdir 0 1 2 3 4 5 6 7 8 9
cd ..
mkdir testing
cd testing
mkdir 0 1 2 3 4 5 6 7 8 9
cd ../../..
python hard_mnist.py


mkdir trained_models/
python mnist_dataset.py

git clone https://github.com/idiap/fullgrad-saliency.git
mv fullgrad-saliency/saliency/ saliency/


mkdir mnist_ups4_outdir/
python distillation.py -dataset mnist -out mnist_ups4_outdir/ -ups 4 --model_path trained_models/hard_mnist_rn34.pth -lr 5000
python inference_distill.py -dataset mnist -out mnist_ups4_outdir/ -ups 4 --model_path mnist_ups4_outdir/fs_1.pth -lr 5000


python pixel_perturbation.py -dataset mnist --model_path trained_models/hard_mnist_rn34.pth -ups 4 --mask_path mnist_ups4_outdir/ --mask_num 1 -p 10 20 50 100
python pixel_perturbation_ours.py -dataset mnist --model_path mnist_ups4_outdir/fs_1.pth -ups 4 --mask_path mnist_ups4_outdir/ --mask_num 1 -p 10 20 50 100

python iou.py --model_path trained_models/hard_mnist_rn34.pth -ups 8 --mask_path mnist_ups8_outdir/ --mask_num 1
