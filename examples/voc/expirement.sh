 CUDA_VISIBLE_DEVICES=0  python ./train_fcn32s.py -g 0 --lr 1.0e-10 --momentum 0.9 --save_ckp fcn32

 CUDA_VISIBLE_DEVICES=1  python train_fcn16s.py -g 2 --lr 1.0e-10 --momentum 0.9 --save_ckp fcn16

 CUDA_VISIBLE_DEVICES=2  python train_fcn8s.py -g 5 --lr 1.0e-8 --momentum 0.9 --save_ckp fcn8

CUDA_VISIBLE_DEVICES=3 python evaluate.py -model_file /home/shlee/workspace/CV/pj2/pytorch-fcn/examples/voc/logs/fcn32s/fcn32s_checkpoint.pth.tar  -g 2