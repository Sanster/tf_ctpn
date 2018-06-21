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
Download pre-trained CTPN model(based on vgg16) from [google drive](https://drive.google.com/open?id=1Cu3qomZFdH_TUkqeFkwtyA6OD_6BRkuR), put it in `output/vgg16/voc_2007_trainval/default`.
Run 
```
python3 tools/demo.py
```

This model is trained on 1080Ti with 50k iterations using this commit 280ee63b2394140c1c0094db8b3329fbb9db21e1.
The final total loss is around 0.35.

# Training
1. Download training dataset from [google drive](https://drive.google.com/open?id=16g1wq2PAqMfDXzim-GB7AK9MBmPIGxQb). This dataset is prepare by the author of [text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn).
According to the author this dataset is from [Multi-lingual scene text detection](http://rrc.cvc.uab.es/?ch=8&com=downloads).
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

# Run on ICDRA15 Incidental Scene Text
```
python3 tools/icdar.py --img_dir=path/to/ICDAR15/incidental_scene_text/ch4_test_images
```

After finish, a submit.zip file will generated in `tools/ICDAR15`, than run the `script.py`

```
cd tools/ICDAR15
# use python2
python script.py -g=gt.zip -s=submit.zip
```

# Some Notes
- Change configs in `./lib/text_connector/text_connect_cfg.py` to match your text detect task.
- Text is very different from 'object' defined in ImageNet. When use tf-slim pre-trained VGG16 model, I need to
make all layer trainable to get good text detect results

# Todo
- [ ] Support ResNet
- [ ] Support MobileNet
