# SLAP: Improving Physical Adversarial Examples with Short-Lived Adversarial Perturbations

Artifacts for "[SLAP: Improving Physical Adversarial Examples with Short-Lived Adversarial Perturbations](https://arxiv.org/abs/2007.04137)" to appear in Usenix Security 2021.

The code for the experiments will come soon.

<p align="center"><img src="https://raw.githubusercontent.com/ssloxford/short-lived-adversarial-perturbations/master/images/slap.gif" width="70%"></p>

## Data and Models
Data and models used in this paper are available in this [release](https://github.com/giuliolovisotto/short-lived-adversarial-perturbations/releases/tag/usenix21).
The `data.zip` folder will contain the following files:
```
.  # data.zip
├── models                    # contains all models
│   ├── yolov3           
│   ├── maskrcnn
│   ├── gtsrbcnn
│   │   │── cvpr18            # taken from cvpr18 paper
│   │   │── cvpr18iyswim      # above with input randomization layer added
│   │   │── usenix21          # architecture from cvpr18 but re-trained
│   │   └── usenix21adv       # above but trained with adversarial loss
│   └── lisacnn               # same subfolders as gtsrbcnn
├── indoor                    # data used in indoors experiment
│   ├── objects               # target objects cutouts
│   ├── profile               # profiling data, used to fit projection model
│   │   │── 120/stop_sign/all_triples.csv  # file used in 120 lux, for stop sign target
│   │   └── ...        
│   └── optimize
│       │── 120/stop_sign/yolov3/_best_projection.png  # projection image used for 120 lux, stop sign, yolov3 model
│       └── ...        
├── outdoor                    # data used in outdoors experiment 
└── datasets                   # image datasets used for re-training
    ├── gtsrb           
    └── lisa
```

## Using this codebase
This repository contains most of the code used in the paper.
We also publish a [**Docker image**](https://hub.docker.com/r/giuliolovisotto/short-lived-adversarial-perturbations/)
containing the execution environment and
[**data and models**](https://github.com/giuliolovisotto/short-lived-adversarial-perturbations/releases/tag/usenix21)
used in this work. 

To use this repository, you will need `docker`, `docker-compose` and a GPU. 
Make sure that the GPU is available within docker containers, (e.g., we had to 
set `"default-runtime": "nvidia"` in `/etc/docker/daemon.json`). 
Most of the code will also work on a CPU once you replace tensorflow-gpu with tensorflow==1.15.3;
if you do not have a GPU you can do this inside the container or rebuild it from a cpu-based image 
(`FROM tensorflow/tensorflow:1.15.3-py3-jupyter`).

Follow these steps:
 
 * `git clone https://github.com/giuliolovisotto/short-lived-adversarial-perturbations.git`
 * `cd short-lived-adversarial-perturbations/`
 * `wget https://github.com/giuliolovisotto/short-lived-adversarial-perturbations/releases/download/usenix21/data.zip`
 the `data.zip` contains models and data necessary to run tests.
 * `unzip data.zip`
 * `docker pull giuliolovisotto/short-lived-adversarial-perturbations:usenix21`
 * `docker-compose up -d`
 * `docker attach slap_container`
 * `nose2` this runs a set of tests which verify that the basic functionalities run correctly.
 
One can check examples of usage by locally opening the notebook
the notebook [example.ipynb](https://localhost:5749)
(the notebook is running inside the container and forwarded on this port).
The password is `slap@usenix21`.

You can also check the content of this notebook online
[example.ipynb](https://github.com/giuliolovisotto/short-lived-adversarial-perturbations/blob/main/code/example.ipynb).

## Contributors
 * [Giulio Lovisotto](https://giuliolovisotto.github.io)
 * [Henry Turner](http://www.cs.ox.ac.uk/people/henry.turner/)

## Acknowledgements

This work was generously supported by a grant from armasuisse and by the Engineering and Physical Sciences Research Council \[grant numbers EP/N509711/1, EP/P00881X/1\].
