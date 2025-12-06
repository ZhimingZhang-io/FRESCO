taskname='linearP'
backbone='resnet18'
pretrain_path='../checkpoints/resnet18_MIMIC_bestZeroShotAll_encoder.pth'


bash ./sub_script/icbeb/sub_icbeb.sh $taskname $backbone $pretrain_path


bash ./sub_script/chapman/sub_chapman.sh $taskname $backbone $pretrain_path


bash ./sub_script/ptbxl/sub_ptbxl.sh $taskname $backbone $pretrain_path
