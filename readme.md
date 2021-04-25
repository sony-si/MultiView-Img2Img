# AI-Research
## [Multi-View Image-to-Image Translation Supervised by 3D Pose](./readme.md)

Authors: Idit Diamant*, Oranit Dror*, Hai Victor Habi, Arnon Netzer

\* equal contribution

Paper: [arxiv 2104.05779](https://arxiv.org/abs/2104.05779)

We address the task of multi-view image-to-image translation for person image generation. The goal is to synthesize photo-realistic multi-view images with pose-consistency across all views. Our proposed end-to-end framework is based on a joint learning of multiple unpaired image-to-image translation models, one per camera view-point. The joint learning is imposed by constraints on the shared 3D human pose in order to encourage the 2D pose projections in all views to be consistent. Experimental results on the CMU-Panoptic dataset demonstrate the effectiveness of the suggested framework in generating photo-realistic images of persons with new poses that are more consistent across all views in comparison to a standard Image-to-Image baseline.


## Install
Installation requirements via pip:
```
pip install requirements.txt
```
Download a 3D pose estimation model from the [learnable-triangulation-pytorch](http://github.com/karfly/learnable-triangulation-pytorch/) github.

##Train
```
python train.py --use_aug --gpu_ids 0,1,2,3 --name train_experiment_name --model cycle_3Dpose_gan
```
##Evaluation
```
python test.py --gpu_id 3 --view view0 --dataroot ./cmu-panoptic/subsets/171026_pose1_pose2/00_00/personA/ --name cmu_171026_pose1_dual_view/view0  --model_suffix "_A" --model test --num_test 6000 --no_dropout
```

## Contribution
If you find a bug or have a question, please create a GitHub issue.

