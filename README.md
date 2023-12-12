# Verifiable Feature Attributions: A Bridge between Post Hoc Explainability and Inherent Interpretability
The script for the full pipeline is at script.sh


## To set up datasets:
Hard MNIST:
Clone https://github.com/jayaneetha/colorized-MNIST, then run `python hard_mnist.py`

Chest X-ray:
Download from https://www.kaggle.com/datasets/paulti/chest-xray-images

CelebA:
Download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html or https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

## To train baseline models (f_b):
```
python mnist_dataset.py
python xray_dataset.py
python celeba_dataset.py
```

## DiET:
```
mkdir mnist_ups4_outdir/
python distillation.py -dataset mnist -out mnist_ups4_outdir/ -ups 4 --model_path trained_models/hard_mnist_rn34.pth -lr 2000
python inference_distill.py -dataset mnist -out mnist_ups4_outdir/ -ups 4 --model_path mnist_ups4_outdir/fs_1.pth -lr 2000
```

## Evaluation:
### Pixel Perturbation:
```
python pixel_perturbation.py -dataset mnist --model_path trained_models/hard_mnist_rn34.pth -ups 4 --mask_path mnist_ups4_outdir/ --mask_num 1 -p 10 20 50 100
python pixel_perturbation_ours.py -dataset mnist --model_path mnist_ups4_outdir/fs_1.pth -ups 4 --mask_path mnist_ups4_outdir/ --mask_num 1 -p 10 20 50 100
```

### IOU:
```
python iou.py --model_path trained_models/hard_mnist_rn34.pth -ups 4 --mask_path mnist_ups4_outdir/ --mask_num 1
```

### Model Faithfulness:
Results reported during distillation (in `distillation.py`)


## Packages required:

- torch
- torchvision
- numpy
- pillow
- pandas
- https://github.com/idiap/fullgrad-saliency
