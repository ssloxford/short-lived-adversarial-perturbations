## Experiment flow as used in the paper
In order to run an experiment end-to-end, you will need: 
  * camera, 
  * projector,
  * lux meter,
  * target object (e.g., stop sign).

The experiments in the paper all have the same steps as follows:
### 1. Setup
All the results for an experiment will be saved in the `data/<ID>` where
`<ID>` is your chosen experiment id.
The following setup steps allow you to setup the rest of the experiment.
 * Setup projector so that it points to the target object, adjust its focus
 and zoom.
 * Create a (good resolution) `.png` image containing the the target object
 with transparent background.
 Save this image in `data/<ID>/objects/<OBJ>.png`
 * Use the lux meter to measure the current ambient light on the target
 object in lux,  `<LUX>`.
 
We report examples of the three objects used in the paper in
`data/test_run/objects`: `stop_sign.png`, `bottle.png`, `give_way.png`

<p align="center">
<img src="https://raw.githubusercontent.com/giuliolovisotto/short-lived-adversarial-perturbations/main/data/test_run/objects/stop_sign.png" width="20%">
<img src="https://raw.githubusercontent.com/giuliolovisotto/short-lived-adversarial-perturbations/main/data/test_run/objects/bottle.png" width="20%">
<img src="https://raw.githubusercontent.com/giuliolovisotto/short-lived-adversarial-perturbations/main/data/test_run/objects/give_way.png" width="20%">
</p>

NB. it is fundamental that the `<OBJ>.png` image has a well defined
transparent fourth channel, as this is used as a mask in later Step 4.

### 2. Run Profiling
In this step one collects all the data necessary to fit the
projection model, i.e., the set of O, S, P used in Equation 1
in the paper.
 * Run `python projector/profile.py -id <ID> -l <LUX> -ob <OBJ>`.
   
Generates a file `/home/data/<ID>/profile/<LUX>/<OBJ>/all_triples.csv` which
contains the resulting triples, i.e., columns 1-3 are the rgb color
values for S, columns 4-6 values for P, columns 7-9 values for O.
```
origin_r,origin_g,origin_b,addition_r,addition_g,addition_b,outcome_r,outcome_g,outcome_b,outcome_std,n_matching_pixels
74,48,50,254,254,0,104,105,46,0.0,1
74,50,53,254,254,127,95,107,100,0.0,1
75,43,45,254,0,127,105,38,92,0.0,1
...
```

We report the collected `all_triples.csv` files used in the paper in
the release TODO. The folder `indoor/` contains the data collected indoor,
while the folder `outdoor/` contains the data collected outdoor.

### 3. Fit Projection Model
This is the projection model optimization (Equation 2 in the paper). 
 * `python projector/projection_model.py -id <ID> -pm <LUX> -ob <OBJ>`.  
 
Generates various files in `/home/data/<ID>/projection_model/<LUX>/<OBJ>`,
the files `projection_model.json` and `params.json` are required by 
Step 4.

In the paper experiments we repeated the optimization
increasing the number of units in the hidden layer
if the average per-channel MSE loss was higher than 0.03.

### 4. Generate Adversarial Example
This is the optimization of the adversarial projection through
the projection model with data augmentation (Equation 3 in the paper). 
 * `python classifiers/optimize.py -id <ID> -n <CLASSIFIER> -pm <LUX> -ob <OBJ>`
 
Generates outputs in `/home/data/<ID>/optimize/<LUX>/<OBJ>/<CLASSIFIER>`:
 * `inputs/` is a visualization of the network inputs,
 * `traindata/` is a visualization of the augmented data used to create
 network inputs,
 * `visualize/` contains images showing the status of the optimization
 at each epoch. 
 
The resulting optimized projection can be found in `_best_projection.png`.

<p align="center">
<img src="https://raw.githubusercontent.com/giuliolovisotto/short-lived-adversarial-perturbations/main/images/used_projections.png" width="60%">
</p>
 
### 5. Project adversarial image and collect videos
In the paper, we shine the `_best_projection.png` image on the target
object, making sure to correct the perspective of the image based on
the projector point-of-view. 
With the image being shown, we collect a set of videos of the target
object from various angles and distances.
<p align="center"><img src="https://raw.githubusercontent.com/giuliolovisotto/short-lived-adversarial-perturbations/main/images/sample_video.gif" width="60%"></p>

### 6. Detect objects
In the paper we engineered our own detector which took input videos
and tracked the location of the stop sign with the help of the user.
We provide in this paper a simpler version of such detector which 
takes a single image in input.
When using `maskrcnn` or `yolov3` one can call such detection like so:
 * `python /home/code/classifiers/detect.py -n0 <CLASSIFIER> -f <INPUT_IMAGE> -o <OUTPUT_FOLDER>`

For `LisaCNN` and `GtsrbCNN` there  are various models to choose from,
specified by the `-n1` argument: 
 * `cvpr18` is the model take from the cvpr18 paper [1],
 * `cvpr18iyswim` is `cvpr18` with the randomization layer of [2],
 * `usenix21` is the same architecture as `cvpr18` but we re-trained it
 from scratch with our train/test/val split and data augmentation.
 * `usenix21adv` is `usenix21` but trained with adversarial learning. 
 
For all of these, you can not simply feed the entire image to the model since they
only take a cutout of the target object, you will have to define a
roi. This can be done either by providing the `--roi` argument, or if you 
have a display available an opencv window will pop up asking you to draw the
roi on the input frame.
 * `python /home/code/classifiers/detect.py -n0 <CLASSIFIER> -n1 <CLF_ID> -f <INPUT_IMAGE> -o <OUTPUT_FOLDER> --roi X Y W H`

You can check `example.ipynb` for an example usage.

[1] - Eykholt, Kevin, et al. **"Robust physical-world attacks on deep learning visual classification."** Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.  
[2] - Xie, Cihang, et al. **"Mitigating adversarial effects through randomization."** arXiv preprint arXiv:1711.01991 (2017).

#### 7. Defences

We have three different defences in the paper:

 * *input_randomization*: is obtained by simply adding new resizing and
 reshaping layers before the image is fed into the network. We provide
 the two models obtained this way with `gtsrbcnn_cvpr18iyswim` and 
 `lisacnn_cvpr18iyswim`,
 * *adversarial learning*: we used the [neural structured learning library](https://www.tensorflow.org/neural_structured_learning)
to re-train our models from scratch with and without adversarial loss.
The four obtained models models are provided in `data/models`:
   - `lisacnn_usenix21`,
   - `gtsrbcnn_usenix21adv`,
   - `lisacnn_usenix21`,
   - `gtsrbcnn_usenix21adv`.
 * *Sentinet*: we implemented Sentinet (`classifiers/defences/sentinet.Sentinet`).

The first two defended models can be used as detectors as shown in above
Step 6, instead check `example.ipynb` for an example usage of Sentinet.