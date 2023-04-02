#!/bin/bash
# full pipeline of VisTracker, assume the runtime env is already activated
seq=$1
recon_name="test-releasev2" # reconstruction save name

echo "Step 1: fit SMPLH to keypoints"
python preprocess/fit_SMPLH_30fps.py -s ${seq} -bs 512

# step 2: smooth and fit again
python smoothnet/smooth_smplt.py --cfg smoothnet/configs/pw3d_spin_3D.yaml --exp_name smplt-srela-w64 -n smoothnet-smpl \
--body_representation smpl-trans --smpl_relative -s ${seq}
python preprocess/fit_SMPLH_smoothed.py -sn smplt-smoothed -s ${seq}
python preprocess/pack_smplt.py -t 1 -m smoothed -s ${seq}

# step 3: render SMPL-T as triplane
python render/render_triplane_nr.py -s ${seq}

# step 4: run SIF-Net
python recon/recon_fit_trivis_full.py tri-vis-l2 -sn test-release -or neural -sr smplt-smoothed-fit -t 1 -bs 64 -tt smooth -neural_only -s ${seq}
# step 4.1: pack recon
python preprocess/pack_recon.py -sn test-release -neural_only -s ${seq}

# step 5: run SmoothNet + HVOP-Net
python smoothnet/smooth_objrot.py --cfg smoothnet/configs/pw3d_spin_3D.yaml --exp_name orot-w64d2 -n smoothnet \
--body_representation obj-rot -neural_pca -or test-release -s ${seq}
python interp/test_cinfill_autoreg.py cmf-k4-lrot -sr smplt-smoothed-fit -or test-release-smooth  -sn smooth-hvopnet -s ${seq}

# step 6: joint optimization and pack results
python recon/recon_fit_trivis_full.py tri-vis-l2 -sr smplt-smoothed-fit -or smooth-hvopnet -sn test-releasev2 -s ${seq}
python preprocess/pack_recon.py -sn ${recon_name} -s ${seq}

# step 7: evaluate, need to specify the split file including all sequences to be evaluated
#python recon/eval/evalvideo_packed.py -split splits/test.json -sn ${recon_name} -m ours -w 300

# step 8: visualization
python render/render_side_comp.py -s1 ${recon_name} -s ${seq}
