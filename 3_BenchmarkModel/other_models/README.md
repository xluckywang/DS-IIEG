Terminal运行时需要cd到classification
tensorboard --logdir=results_train/Ori/MTD/vgg16
tensorboard --logdir=results_train/Ori/MTD/vgg16 --port=6007

运行train.py训练数据集

txt_annotation.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py

visual_tsne.py (运行前 txt_annotation_tsne.py)
visual_cam.py


