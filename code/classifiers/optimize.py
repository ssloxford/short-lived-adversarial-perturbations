import os
import json
import shutil
import argparse
from argparse import Namespace

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import classifiers_utils as clfutils
from classifiers.yolov3.optimizer import Yolov3Optimizer
from classifiers.gtsrbcnn.optimizer import GtsrbOptimizer
from classifiers.lisacnn.optimizer import LisaOptimizer
from classifiers.maskrcnn.optimizer import MaskRCNNOptimizer
from classifiers import generate_traindata
import numpy as np
import sys
from classifiers import slap_graph_def
import tensorflow as tf
import cv2
import skimage.io

from collections import OrderedDict

OPTIMIZER_MAP = {
    "yolov3": Yolov3Optimizer,
    "gtsrbcnn_cvpr18": GtsrbOptimizer,
    "lisacnn_cvpr18": LisaOptimizer,
    "maskrcnn": MaskRCNNOptimizer
}


def plot_imgs(imgs, savepath, ncols=10):
    n_imgs = len(imgs)
    nrows = n_imgs//ncols + 1
    if n_imgs % ncols == 0:
        nrows-=1
    fig, ax = plt.subplots(nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
    for i, img in enumerate(imgs):
        y = i % ncols
        x = int(i/ncols)
        ax[x, y].imshow(img)
        ax[x, y].set_xticks([])
        ax[x, y].set_yticks([])
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)


def save_intermediate(script_args, projections, opt_folder, epoch, pnorm_s, de_s, tv_s, patches, clf_input):
    projections = np.array(projections)
    patches = np.array(patches, dtype=object)
    np.save(os.path.join(opt_folder, "projections.npy"), projections)
    np.save(os.path.join(opt_folder, "patches.npy"), patches)
    cv2.imwrite(os.path.join(opt_folder, "_best_projection.png"), (projections[-1][..., ::-1] * 255).astype(int))
    cv2.imwrite(os.path.join(opt_folder, "_best_patch.png"), (patches[-1][..., ::-1] * 255).astype(int))

    for i, image in enumerate(clf_input):
        cv2.imwrite(os.path.join(opt_folder, "inputs", "_%d.png" % i), (image[..., ::-1] * 255).astype(int))

    opt_params = vars(script_args)
    opt_params["r"] = {}
    opt_params["r"]["pnorm_mean"] = float(pnorm_s.mean())
    opt_params["r"]["detection_score_mean"] = float(de_s.mean())
    opt_params["r"]["total_variation"] = float(tv_s)
    opt_params["r"]["epoch"] = int(epoch)
    json.dump(opt_params, open(os.path.join(opt_folder, "params.json"), "w"), indent=2)
    return True


def join_namespaces(namespaces_list):
    output = {}
    for namespace in namespaces_list:
        ns_dict = vars(namespace)
        for key, value in ns_dict.items():
            if value is not None or key not in output:
                output.update({key: value})
    return Namespace(**output)


def slap_run(gen, slappy, opt, t_to_eval, b_start, b_end):
    feed_dict = {
        slappy["input_bg_imgs"]: gen.batch_bg_imgs[b_start:b_end],
        slappy["traffic_sign_imgs"]: gen.batch_ts_imgs[b_start:b_end],
        slappy["box_coords"]: gen.boxes[b_start:b_end],
        slappy["projection_shift"]: gen.shifts[b_start:b_end],
        slappy["perspective_transform_h"]: gen.perspective_transform_h[b_start:b_end],
        slappy["perspective_transform_v"]: gen.perspective_transform_v[b_start:b_end],
        slappy["perspective_left_or_right"]: gen.perspective_left_or_right[b_start:b_end],
        slappy["perspective_up_or_down"]: gen.perspective_up_or_down[b_start:b_end],
        slappy["gauss_blur_std"]: gen.gaussian_blur_std[b_start:b_end],
        slappy["gauss_blur_fs"]: gen.gaussian_blur_fs[b_start:b_end],
        slappy["brightness_variation_pre"]: gen.brightnesses_pre[b_start:b_end],
        slappy["brightness_variation_post"]: gen.brightnesses_post[b_start:b_end],
        slappy["projection_in_box_mask"]: gen.proj_in_box_mask,
        slappy["rotation_pad"]: gen.rotation_pad,
        slappy["rotation_angle"]: gen.rotation_angles[b_start:b_end],
        slappy["channel_offset"]: gen.channel_offset[b_start:b_end],
        slappy["traffic_sign_box_pad"]: gen.traffic_sign_box_pad[b_start:b_end]}
    feed_dict.update(opt.mandatory_feeds())

    to_eval = [v for k, v in t_to_eval.items()]
    ret_list = sess.run(to_eval, feed_dict=feed_dict)
    ret_dict = OrderedDict()
    for i, (key, value) in enumerate(t_to_eval.items()):
        ret_dict[key] = ret_list[i]
    return ret_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate adversarial examples for target model using with expectation"
                                                 "over transformation and projection model.")
    parser.add_argument("-n", "--detector_name", type=str, required=True,
                        choices=["yolov3", "gtsrbcnn_cvpr18", "lisacnn_cvpr18", "maskrcnn"])
    parser.add_argument("-id", "--experiment_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("-pm", "--proj_model_lux", type=str, required=True, help="Projection model lux")
    parser.add_argument("-ob", "--object_id", type=str, required=True, help="Object ID")
    parser.add_argument("--n_epochs", type=int, required=False, help="Number of epochs for optimization")

    # merge args, arguments passed to the script will override defauls in params.yaml
    args1 = parser.parse_args()
    args2 = clfutils.get_optimizer_args(args1.detector_name.split("_")[0], args1.object_id)
    args = join_namespaces([args2, args1])

    this_script_name = os.path.basename(__file__).split(".")[0]
    experiment_id, pm_lux, object_id = args.experiment_id, args.proj_model_lux, args.object_id
    experiment_folder = os.path.join("/home/data/%s" % experiment_id)

    # fetch projection model
    pmodel_folder = os.path.join(experiment_folder, "projection_model", pm_lux, object_id)
    proj_model_params = json.load(open(os.path.join(pmodel_folder, "params.json"), "r"))
    proj_model_dict = json.load(open(os.path.join(pmodel_folder, "projection_model.json"), "r"))

    # create output folder
    optimizer_folder = os.path.join(experiment_folder, this_script_name, pm_lux, object_id, args.detector_name)
    shutil.rmtree(optimizer_folder, ignore_errors=True)
    os.makedirs(os.path.join(optimizer_folder, "traindata"), exist_ok=True)
    os.makedirs(os.path.join(optimizer_folder, "visualize"), exist_ok=True)
    os.makedirs(os.path.join(optimizer_folder, "inputs"), exist_ok=True)

    backgrounds_path = os.path.join(experiment_folder, "backgrounds")
    bg_imgs = generate_traindata.get_bg_imgs(backgrounds_path)

    object_img_fpath = os.path.join(experiment_folder, "objects", object_id+".png")
    assert os.path.isfile(object_img_fpath), "%s.png does not exist" % object_id
    traffic_sign_imgs = [skimage.io.imread(object_img_fpath).astype(np.float32)/255.0]  # float rgb

    with tf.Graph().as_default() as slap_graph:
        # add our instructions
        slappy = slap_graph_def.add_slap_instructions(args.grid_size, args.grid_size, batch_size=args.batch_size,
            pm_l1=proj_model_params["number_of_neurons_in_l1"], resize_method=args.interpolation_method,
            init=args.projection_init_value, bypass_pm=False)

        # import the meta graph and link output to applied_perturbation
        opt = OPTIMIZER_MAP[args.detector_name]

        classif_input = opt.slapped_input_to_network_input(slappy["box_coords_padded"], slappy["slapped_input"], **vars(args))
        saver = tf.compat.v1.train.import_meta_graph(args.meta_graph_path, input_map={opt.get_input_tensor_name(): classif_input})
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(graph=slap_graph, config=config) as sess:
            # load weights
            optimizer = opt(saver, sess, **vars(args))

            loss_function, total_variation_reg, det_scores_sum, pnorm = slap_graph_def.make_loss_function(
                slappy, optimizer.scores, args.index_of_misdetection_class, args.batch_size, args.pnorm_mult,
                args.total_variation_mult, args.target_class_mult, target_class=args.target_class_index)

            # define optimizer(s)
            with tf.compat.v1.variable_scope('adam/complete'):
                adam_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=args.adam_learning_rate, beta1=args.adam_beta1,
                    beta2=args.adam_beta2).minimize(loss_function, var_list=[slappy["projection_base"]])

            # initialize perturbation plus adam opt variables plus projection model
            sess = slap_graph_def.slap_initialize_vars(sess, proj_model_dict, slappy)

            # optimization start
            current_epoch = 0

            # prepare things you want to save
            projections, patches = [], []

            tensors_to_evaluate = OrderedDict({
                "adam": adam_opt, "tv": total_variation_reg, "det": det_scores_sum, "pnorm": pnorm,
                "masked_proj": slappy["masked_projection"], "proj_r_s": slappy["projs_r_s"],
                "proj_in_box_mask_r_s": slappy["projs_in_box_mask_r_s"], "stop_signs_imgs_r": slappy["stop_sign_imgs_r"],
                "stop_sign_in_box_masks": slappy["stop_sign_in_box_masks"],
                "stop_signs_imgs_r_bv": slappy["stop_sign_imgs_r_bv"],
                "proj_model_outs": slappy["proj_model_outs"], "stop_signs_pts": slappy["stop_signs_pts"],
                "proj_model_deltas": slappy["proj_model_deltas"], "patches": slappy["patches"],
                "clf_input": classif_input})

            # main loop
            while current_epoch < args.n_epochs:

                # generate training data
                gen = generate_traindata.get_traindata_v2(bg_imgs, traffic_sign_imgs, bypass_pm=False, **vars(args))
                gen = Namespace(**gen)

                ce_print_name = str(current_epoch).zfill(5)
                det_agg, pnorm_agg, train_plot = [], [], []

                for batch_i in range(args.batches_before_checkpoint):
                    b_start, b_end = batch_i*args.batch_size, (batch_i+1)*args.batch_size
                    r = slap_run(gen, slappy, optimizer, tensors_to_evaluate, b_start, b_end)
                    sys.stdout.write("\repoch: %s/%s, batch %d/%d, tv: %.5f, max_ss: %.5f, pnorm: %.5f" % (
                        ce_print_name, str(args.n_epochs).zfill(5), batch_i + 1, args.batches_before_checkpoint,
                        r["tv"], r["det"], r["pnorm"]))
                    sys.stdout.flush()

                    det_agg.append(r["det"])
                    pnorm_agg.append(r["pnorm"])

                    if r["clf_input"].ndim == 3:
                        r["clf_input"] = r["clf_input"][np.newaxis, ...]

                    train_plot.append(
                        cv2.resize(
                            np.clip((optimizer.adjust_input_for_plot(r["clf_input"][0]) * 255).astype(np.uint8), 0, 255),
                            (416, 416)))

                det_agg = np.array(det_agg)
                if args.redo_hard_batches > 0.0:
                    indexes_of_hard_batches = np.sort(np.argsort(det_agg)[::-1][:int(det_agg.shape[0]*args.redo_hard_batches)])
                    for _ in range(1):
                        for hard_batch_i in indexes_of_hard_batches:
                            b_start, b_end = hard_batch_i*args.batch_size, (hard_batch_i+1)*args.batch_size
                            r = slap_run(gen, slappy, optimizer, tensors_to_evaluate, b_start, b_end)
                            sys.stdout.write("\rhb: epoch: %s/%s, batch %d/%d, tv: %.5f, max_ss: %.5f, pnorm: %.5f" % (
                                ce_print_name, str(args.n_epochs).zfill(5), hard_batch_i + 1, args.batches_before_checkpoint, r["tv"], r["det"], r["pnorm"]))
                        sys.stdout.flush()
                        det_agg[hard_batch_i] = r["det"]
                        pnorm_agg[hard_batch_i] = r["pnorm"]

                if r["clf_input"].ndim == 3:
                    r["clf_input"] = r["clf_input"][np.newaxis, ...]

                pnorm_agg = np.array(pnorm_agg)
                projections.append(r["masked_proj"])
                patches.append(clfutils.crop_black(r["patches"][-1]))

                sys.stdout.write("\repoch: %s/%s, tv: %.5f, det_score(avg): %.5f, pnorm(avg): %.5f\n" % (
                    ce_print_name, str(args.n_epochs).zfill(5), r["tv"], det_agg.mean(), pnorm_agg.mean()))
                sys.stdout.flush()

                current_epoch += 1

                # save some intermediate results
                classif_input_ = optimizer.adjust_input_for_plot(r["clf_input"])
                save_intermediate(args, projections, optimizer_folder, current_epoch, pnorm_agg, det_agg, r["tv"], patches, classif_input_)
                plot_imgs(train_plot, os.path.join(optimizer_folder, "traindata", "%s.png" % ce_print_name), ncols=10)
                np.save(os.path.join(optimizer_folder, "traindata", "train_imgs.npy"), np.array(train_plot))

                clfutils.myimgshow(
                    [r["stop_signs_imgs_r_bv"][0], r["masked_proj"], r["proj_model_outs"][0], r["stop_signs_pts"][0],
                     classif_input_[0], r["proj_model_deltas"][0], r["patches"][0]],
                    titles=["base ss", "projection", "proj_model_outs", "stop_signs_pt_", "slapped_input_", "pmd_", "patch_"],
                    fname=os.path.join(optimizer_folder, "visualize", "%s.png" % ce_print_name))

    sys.stdout.write("Everthing is saved\n")
