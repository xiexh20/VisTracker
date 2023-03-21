# VisTracker
Official implementation for the CVPR'23 paper 

Train a model:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_launch.py -en tri-vis-l2
```

