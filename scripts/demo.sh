#!/bin/bash
# define colors for cmd output
GREEN=$'\e[0;32m'
RED=$'\e[0;31m'
NC=$'\e[0m'

# full pipeline of VisTracker, assume the runtime env is already activated
seq=$1
recon_name="test-releasev2" # reconstruction save name

set -e

echo "${RED} VisTracker demo step 1/7: fit SMPLH to keypoints ${NC}"
python preprocess/fit_SMPLH_30fps.py -s ${seq} -bs 512

echo "${RED} VisTracker demo step 2/7: smooth and fit again ${NC}"
python smoothnet/smooth_smplt.py --cfg smoothnet/configs/pw3d_spin_3D.yaml --exp_name smplt-srela-w64 -n smoothnet-smpl \
--body_representation smpl-trans --smpl_relative -s ${seq}
python preprocess/fit_SMPLH_smoothed.py -sn smplt-smoothed -s ${seq}
python preprocess/pack_smplt.py -t 1 -m smoothed -s ${seq}

echo "${RED} VisTracker demo step 3/7: render SMPL-T as triplane ${NC}"
python render/render_triplane_nr.py -s ${seq}

echo "${RED} VisTracker demo step 4/7: run SIF-Net ${NC}"
python recon/recon_fit_trivis_full.py tri-vis-l2 -sn test-release -or neural -sr smplt-smoothed-fit -t 1 -bs 64 -tt smooth -neural_only -s ${seq}
# step 4.1: pack recon
python preprocess/pack_recon.py -sn test-release -neural_only -s ${seq}

echo "${RED} VisTracker demo step 5/7: run SmoothNet + HVOP-Net ${NC}"
python smoothnet/smooth_objrot.py --cfg smoothnet/configs/pw3d_spin_3D.yaml --exp_name orot-w64d2 -n smoothnet \
--body_representation obj-rot -neural_pca -or test-release -s ${seq}
python interp/test_cinfill_autoreg.py cmf-k4-lrot -sr smplt-smoothed-fit -or test-release-smooth  -sn smooth-hvopnet -s ${seq}

echo "${RED} VisTracker demo step 6/7: joint optimization and pack results ${NC}"
python recon/recon_fit_trivis_full.py tri-vis-l2 -sr smplt-smoothed-fit -or smooth-hvopnet -sn test-releasev2 -s ${seq}
python preprocess/pack_recon.py -sn ${recon_name} -s ${seq}

# step 7: evaluate, need to specify the split file including all sequences to be evaluated
#python recon/eval/evalvideo_packed.py -split splits/test.json -sn ${recon_name} -m ours -w 300

echo "${RED} VisTracker demo step 7/7: visualization ${NC}"
python render/render_side_comp.py -s1 ${recon_name} -s ${seq}
