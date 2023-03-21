# VisTracker
Official implementation for the CVPR'23 paper 

Train a model:
```shell
python -m torch.distributed.launch --nproc_per_node=NUM_GPU --master_port 6789 --use_env train_launch.py -en tri-vis-l2
```

Train  motion infill model:
```shell
python -m torch.distributed.launch --nproc_per_node=NUM_GPU --master_port 6787 --use_env train_mfiller.py -en cmf-k4-lrot
```

