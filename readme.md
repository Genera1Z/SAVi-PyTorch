# SAVi-PyTorch -- Unofficial But 3x Faster Training @ Better Performance

## About This Project

Reimplemented by referring to the official code [https://github.com/google-research/slot-attention-video](https://github.com/google-research/slot-attention-video).

The train/eval performance is surely even a bit better under 10 random seeds.

By contrast, the official implementation (shit code) is difficult to setup environment and difficult to debug and difficult to modify for academic experiments.

![train/val curves](/visual.png "train/val curves")
Figure: SAVi-small on MOVi-A.

## Features

- **3x faster training** The model can be trained 3 times faster compared with the official implementation (20 hours -> 7 hours @ GPU 3080 | batch_size=4).
- **5x less I/O load** The dataset MOVi-A is compressed by 5 times (59 GB -> 11 GB), then I/O overhead can be lowered greatly and RAM-disk can be utilized.
- **fp16 train and val** Auto-mixed precision training (fp16) is enabled, then larger batch size can be used and the actual GPU FLOPS can be much larger.
- **config-based experiment** For experiment, the modelling, datasets and learning schemes can all be constructed with a single and concise config file.

## TODO

- Support both medium/large SAVi models
- Support all MOVi dataset subsets

## How to Use

- install requirements: ```pip install -r requirements.txt```
- convert original MOVi-A into LMDB format: ```python datum.py```
- run train and eval: ```python main.py``` or ```sh run.sh```
- visualize the train/val curves: ```python analyze.py```

## About

I am now working on object-centric learning problems. If you have any challenging problems or ideas about this please do not hesitate to contact me.
- WeChat: Genera1Z
- email: rongzhen.zhao@aalto.fi
