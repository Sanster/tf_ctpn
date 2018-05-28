# tf_ctpn

This is a practice project to learn how CTPN works. Most of codes in this project are adapted from
[tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) and [text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)

- CTPN paper: https://arxiv.org/abs/1609.03605
- CTPN source: https://github.com/tianzhi0549/CTPN

# Setup
Install dependencies:
```
pip3 install -r requirements.txt
```

Build Cython part for both demo and training.
```
cd lib/
make clean
make
```

# Quick start
Download pre-trained CTPN model(based on vgg16) from [here]()(Coming soon..), put it in `output/vgg16/voc_2007_trainval/default`.
Run 
```
python3 tools/demo.py
```

This model is trained on 1080Ti with 50k iterations. The finally loss is around ...

# Training
1. Download dataset prepare by the author of [text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn), 
This dataset is from [Multi-lingual scene text detection](http://rrc.cvc.uab.es/?ch=8&com=downloads). 
See [text-detection-ctpn#issues97](https://github.com/eragonruan/text-detection-ctpn/issues/97)
Put dataset in `./data/VOCdevkit2007/VOC2007`

1. Download pre-trained slim vgg16 model from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
Put the pretrained_models in `./data/pretrained_model`

1. Start training
```
python3 tools/trainval_net.py
```
The output checkpoint file will be saved at `./output/vgg16/voc_2007_trainval/default`

1. Start tensorboard
```
tensorboard --logdir=./tensorboard
```

# Todo
- [ ] Support ResNet
- [ ] Support MobileNet

# Some Notes
- Text is very different from 'object' defined in ImageNet. When use tf-slim pre-trained VGG16 model, I need to
make all layer trainable to get good text detect results
