# VisTracker (CVPR'23)
#### Official implementation for the CVPR 2023 paper: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera

[[ArXiv]](https://arxiv.org/abs/2303.16479) [[Project Page]](http://virtualhumans.mpi-inf.mpg.de/VisTracker/)

<p align="left">
<img src="https://datasets.d2.mpi-inf.mpg.de/cvpr23vistracker/teaser.png" alt="teaser" width="512"/>
</p>

## Please also check our old ECCV'22 work CHORE [here](https://github.com/xiexh20/CHORE). 

## Contents 
1. [Dependencies](#dependencies)
2. [Dataset preparation](#dataset-preparation)
3. [Run demo](#run-demo)
4. [Training](#training)
5. [Evaluation](#evaluation)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)
6. [License](#license)

## Dependencies
The code is tested with `torch 1.6, cuda10.1, debian 11`. The environment setup is the same as CHORE, ECCV'22. Please follow the instructions [here](https://github.com/xiexh20/CHORE#dependencies). 


## Dataset preparation
We work on the extended BEHAVE dataset, to have the dataset ready, you need to download some files and run some processing scripts to prepare the data. All files are provided in [this webpage](https://virtualhumans.mpi-inf.mpg.de/behave/license.html). 

1. Download the video files: [color videos of test sequences](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/video/date03_color.tar), [frame time information](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/video/date03_time.tar). 
2. Extract RGB images: follow [this script](https://github.com/xiexh20/behave-dataset#generate-images-from-raw-videos) from BEHAVE dataset repo to extract RGB images. Please enable `-nodepth` tag to extract RGB images only. Example: `python tools/video2images.py /BS/xxie-3/static00/rawvideo/Date03/Date03_Sub03_chairwood_hand.0.color.mp4 /BS/xxie-4/static00/behave-fps30/ -nodepth`
3. Download human and object masks: [masks for all test sequences](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/masks/masks-date03.tar). Download and unzip them into one folder. 
4. Rename the mask files to follow the BEHAVE dataset structure: `python tools/rename_masks.py -s SEQ_FOLDER -m MASK_ROOT` Example: `python tools/rename_masks.py -s /BS/xxie-4/static00/behave-fps30/Date03_Sub03_chairwood_hand -m /BS/xxie-5/static00/behave_release/30fps-masks-new/`
5. Download [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [FrankMocap](https://github.com/facebookresearch/frankmocap) detections: [packed data for test sequences](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-packed-test-seqs.zip)
6. Process the packed data to BEHAVE dataset format: `python tools/pack2separate.py -s SEQ_FOLDER -p PACKED_ROOT`. Example: `python tools/pack2separate.py -s /BS/xxie-4/static00/behave-fps30/Date03_Sub03_chairwood_hand -p /scratch/inf0/user/xxie/behave-packed`

## Run demo 
You can find all the commands of the pipeline in `scripts/demo.sh`. To run it, you need to download the pretrained models from [here](https://datasets.d2.mpi-inf.mpg.de/cvpr23vistracker/models.zip) and unzip them in the folder `experiments`. 

Also, the dataset files should be prepared as described above. 

Once done, you can run the demo for one sequence simply by:
```shell
bash scripts/demo.sh SEQ_FOLDER 
```
example: `bash scripts/demo.sh /BS/xxie-4/static00/test-seq/Date03_Sub03_chairwood_hand`

It will take around 6~8 hours to finish a sequence of 1500 frames (50s). 

Tips: the runtime bottlenecks are the SMPL-T pre-fitting (step 1-2) and joint optimization (step 6) in `scripts/demo.sh`. If you have a cluster with multiple GPU machines, you can run multiple sequences/jobs in parallel by specifying the `--start` and `--end` option for these commands. This will separate one long sequence into several chunks and each job only optimizes the chunk specified by start and end frames. 

## Training 
Train a SIF-Net model:
```shell
python -m torch.distributed.launch --nproc_per_node=NUM_GPU --master_port 6789 --use_env train_launch.py -en tri-vis-l2
```
Note that to train this model, you also need to prepare the GT registrations (meshes) in order to run online boundary sampling during training. We provide an example script to save SMPL and object meshes from packed parameters:
`python tools/pack2separate_params.py -s SEQ_FOLDER -p PACKED_PATH`, similar to `tools/pack2separate.py`. The packed training data for this can be downloaded from [here (part1)](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-packed-train-seqs-p1.zip) and [here (part2)](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-packed-train-seqs-p2.zip)

In addition, the split files, frame times and visibility information should also be downloaded from [here](https://datasets.d2.mpi-inf.mpg.de/cvpr23vistracker/behave-splits.zip) and extracted in the subfolder `splits`. 

Train  motion infill model:
```shell
python -m torch.distributed.launch --nproc_per_node=NUM_GPU --master_port 6787 --use_env train_mfiller.py -en cmf-k4-lrot
```
For this, you need to specify the path to all packed GT files downloaded from the link mentioned above. i.e.: [train part1](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-packed-train-seqs-p1.zip), [train part 2](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-packed-train-seqs-p2.zip), [test seqs](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-packed-test-seqs.zip).  


## Evaluation
```shell
python recon/eval/evalvideo_packed.py -split splits/behave-test-30fps.json -sn RECON_NAME -m ours -w WINDOW_SIZE
```
where `RECON_NAME` is your own save name for the reconstruction, and `WINDOW_SIZE` is the alignment window size (main paper Sec. 4). `WINDOW_SIZE=1` is equivalent to the evaluation used by CHORE. 

## Citation
If you use our code, please cite:
```bibtex
@inproceedings{xie2023vistracker,
title = {Visibility Aware Human-Object Interaction Tracking from Single RGB Camera},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Pons-Moll, Gerard },
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    month={June}, 
    year={2023} 
}
```
If you use BEHAVE dataset, please also cite:
```bibtex
@inproceedings{bhatnagar22behave,
    title = {BEHAVE: Dataset and Method for Tracking Human Object Interactions},
    author={Bhatnagar, Bharat Lal and Xie, Xianghui and Petrov, Ilya and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {jun},
    organization = {{IEEE}},
    year = {2022},
    }
```

## Acknowledgements 
This project leverages the following excellent works, we thank the authors for open-sourcing their code: 

[FrankMocap](https://github.com/facebookresearch/frankmocap)

[Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

[SmoothNet](https://ailingzeng.site/smoothnet)

[Conditional motion infilling](https://github.com/jihoonerd/Conditional-Motion-In-Betweening)

[Interactive segmentation](https://github.com/SamsungLabs/ritm_interactive_segmentation)

[Video segmentation](https://github.com/hkchengrex/MiVOS)

[DetectronV2](https://github.com/facebookresearch/detectron2)




## License
Copyright (c) 2023 Xianghui Xie, Max-Planck-Gesellschaft

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **Visibility Aware Human-Object Interaction Tracking from Single RGB Camera** paper in documents and papers that report on research using this Software.






