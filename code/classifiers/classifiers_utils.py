import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import argparse
import yaml
import random


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


def scale_box_to_org_size(boxes, width_ori, height_ori, tar_width=416, tar_height=416):
    # rescale the coordinates to the original image
    boxes[:, 0] *= (width_ori / float(tar_width))
    boxes[:, 2] *= (width_ori / float(tar_width))
    boxes[:, 1] *= (height_ori / float(tar_height))
    boxes[:, 3] *= (height_ori / float(tar_height))
    return boxes


def get_optimizer_args(opt_name, object_id, yamlfile="/home/code/classifiers/params.yaml"):
    params = yaml.load(open(yamlfile, "r"), Loader=yaml.FullLoader)
    # basic optimizer args
    args = params["base"]
    # optimizer specific base args
    args.update(params[opt_name]["_base"])

    if "object_specific" in params[opt_name] and object_id in params[opt_name]["object_specific"]:
        # optimizer+object specific args
        args.update(params[opt_name]["object_specific"][object_id])
    args = argparse.Namespace(**args)
    return args


def crop_black(img):
    row_sum = img.sum(axis=1)  # (h, nc)
    col_sum = img.sum(axis=0)  # (w, nc)
    non_black_cols = ~np.all(col_sum == 0, axis=1)  # (w, )
    non_black_rows = ~np.all(row_sum == 0, axis=1)  # (h, )
    cropped = np.copy(img)[non_black_rows, ...]
    cropped = cropped[:, non_black_cols]
    return cropped


def get_all_imgpaths_in_folder(folder, format=None):
    filenames = os.listdir(folder)
    if format is not None:
        filenames = list(filter(lambda x: x.split(".")[-1] == format, filenames))
    filepaths = list(sorted(map(lambda x: os.path.join(folder, x), filenames)))
    return filepaths


def myimgshow(imgs, titles=[], fname="here.jpg", verbose=False):
    n = len(imgs)
    assert n<10
    titles = ["" if len(titles) == 0 else titles[i] for i in range(n)]
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(3*n, 3))

    for i, img in enumerate(imgs):
        if verbose:
            a, b = np.histogram(img.flatten())
            print(titles[i], a)
            print(titles[i], b)
            print(titles[i], img.shape, img.min(), img.max(), img.dtype)
        # print(img.shape)
        if img.ndim==2:
            img_ = img[..., np.newaxis]
            img = np.concatenate((img_, img_, img_), axis=2)

        ax[i].set_title(titles[i])
        ax[i].imshow(img)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()
    plt.close()
    return True


def read_image_opencv(image_path, read_flag=cv2.IMREAD_COLOR, output_shape=None, divide=True, broadcast=False):
    img = cv2.imread(image_path, read_flag)
    height_ori, width_ori = img.shape[:2]
    if output_shape is not None:
        assert type(output_shape) == tuple and len(output_shape) == 2, "Feed me right"
        img = cv2.resize(img, tuple(output_shape))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.int32)
    if divide:
        img = img.astype(np.float32)
        img = img / 255.
    if broadcast:
        img = img[np.newaxis, :]
    return img, height_ori, width_ori