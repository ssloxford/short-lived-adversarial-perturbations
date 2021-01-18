import numpy as np
import cv2
import os


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


def get_bg_imgs(folder, format="jpg"):
    bg_imgs = []
    files = os.listdir(folder)
    imgs_names = filter(lambda x: x[-4:] == ".%s" % format, files)
    for img_name in imgs_names:
        img_path = os.path.join(folder, img_name)
        bg_img, _, _ = read_image_opencv(img_path, divide=True, output_shape=(416, 416))
        bg_imgs.append(bg_img)
    return np.array(bg_imgs)


def get_ellipse_mask(h, w):
    ellipse_mask = np.zeros((int(h), int(w)), dtype=float)
    nrow = ellipse_mask.shape[0]
    ncol = ellipse_mask.shape[1]
    for i in range(ellipse_mask.shape[0]):
        for j in range(ellipse_mask.shape[1]):
            if (i-nrow/2)**2 / (nrow/2)**2 + (j-ncol/2)**2/(ncol/2)**2 <= 1.2:
                ellipse_mask[i, j] = 1

    return ellipse_mask


def slap_angle_to_pt_h(angle_h):
    resulting_w = np.sin(np.radians(90 - np.abs(angle_h))) * 1.0
    return [1 / resulting_w, 0, 0, 0, 1, 0, 0, 0]


def slap_angle_to_pt_v(angle_v):
    resulting_h = np.sin(np.radians(90 - np.abs(angle_v))) * 1.0
    return [1, 0, 0, 0, 1 / resulting_h, 0, 0, 0]


def get_horizontal_perspective_transformer(img_size, angle, coef_quotient=150):
    coef = img_size / coef_quotient
    b0 = -(coef * angle / 100.0)*1.5
    c1 = -coef * angle / 10000.0
    return [1.0, 0, 0, b0, 1, 0, c1, -0.000]


def get_vertical_perspective_transformer(img_size, angle, coef_quotient=150):
    coef = img_size / coef_quotient
    a1 = -(coef * angle / 100.0)*.66
    c2 = -coef * angle / 10000.0
    return [1.0, a1, 0, 0, 1, 0, 0, c2]


def get_traindata_v2(bg_imgs, traffic_signs_cut, bypass_pm=False, **kwargs):

    from argparse import Namespace
    cfg = Namespace(**kwargs)

    assert bg_imgs.dtype == "float32", bg_imgs.dtype
    assert bg_imgs.shape[1] == 416, bg_imgs.shape
    assert bg_imgs.shape[2] == 416, bg_imgs.shape
    assert bg_imgs.min() >= 0, bg_imgs.min()
    assert bg_imgs.max() <= 1.0, bg_imgs.max()

    generated = {}
    samples_no = cfg.batches_before_checkpoint*cfg.batch_size

    # pick bg imgs for the data
    batch_bg_imgs_i = np.arange(0, bg_imgs.shape[0])
    batch_bg_imgs = np.array([bg_imgs[j] for j in np.random.choice(batch_bg_imgs_i, samples_no)])

    # boxes and shifts
    boxes = []
    shifts = []

    widths = (cfg.stop_sign_min_dim + np.random.rand(samples_no) * (cfg.stop_sign_max_dim-cfg.stop_sign_min_dim)).astype(int)
    asp_ratio_min_w, asp_ratio_min_h = list(map(int, cfg.camera_aspect_ratio_min.split(":")))
    asp_ratio_max_w, asp_ratio_max_h = list(map(int, cfg.camera_aspect_ratio_max.split(":")))
    mult_min = asp_ratio_min_w/asp_ratio_min_h
    mult_max = asp_ratio_max_w/asp_ratio_max_h
    rand_m = mult_min + np.random.rand(samples_no) * (mult_max - mult_min)
    heights = np.clip((widths * rand_m).astype(int), 1, 415)

    for h, w in zip(heights, widths):
        _x0 = np.random.randint(0, 416 - w)
        _y0 = np.random.randint(0, 416 - h)
        boxes.append([_x0, _y0, _x0 + w, _y0 + h])
        max_pixel_shift = int(max(w, h) * cfg.pixel_shift_ratio)
        shift_direction = np.random.rand(2)*2-1  # \in [-1, 1]
        sample_shift = (shift_direction*max_pixel_shift).astype(int)  # (2, )
        shifts.append(sample_shift)

    shifts = np.array(shifts).reshape(-1, 2)  # (n, 2)
    boxes = np.array(boxes)  # (n, 4)

    # projection in box mask
    padding = int(cfg.stop_sign_min_dim * cfg.projection_padding_ratio)
    if padding % 2 == 1:
        padding = padding + 1
    assert padding % 2 == 0

    # prepare the traffic signs in input
    available_ts = []
    for traffic_sign_img in traffic_signs_cut:
        assert traffic_sign_img.dtype == "float32", traffic_sign_img.dtype
        assert traffic_sign_img.shape[2] == 4, traffic_sign_img.shape
        assert traffic_sign_img.min() >= 0.0, traffic_sign_img.min()
        assert traffic_sign_img.max() <= 1.0, traffic_sign_img.max()
        _resized = cv2.resize((traffic_sign_img * 255).astype(np.uint8), (256, 256), cv2.INTER_CUBIC)
        _resized = _resized.astype(float) / 255.0
        available_ts.append(_resized)

    batch_ts_imgs_i = np.arange(0, len(available_ts))
    batch_ts_imgs = np.array([available_ts[j] for j in np.random.choice(batch_ts_imgs_i, cfg.batches_before_checkpoint * cfg.batch_size)])

    if bypass_pm:
        patch_mask = np.zeros_like(available_ts[0][:, :, 0])[..., np.newaxis]  # (h, w, 1)
        h, w, _ = available_ts[0].shape
        ref_h, ref_w = int(h * .675), int(w * .28)

        actual_h, actual_w = int(h * .3), int(w * .45)
        ellipse_mask = get_ellipse_mask(actual_h, actual_w)[..., np.newaxis]
        patch_mask[ref_h:ref_h+ellipse_mask.shape[0], ref_w:ref_w+ellipse_mask.shape[1], :] = ellipse_mask

        proj_in_box_mask = cv2.resize((patch_mask*255).astype(np.uint8),
                                      (cfg.grid_size - padding, cfg.grid_size - padding), interpolation=cv2.INTER_CUBIC)  # (grid_size, grid_size, 1)
        proj_in_box_mask[proj_in_box_mask < 255] = 0
        proj_in_box_mask = np.pad(proj_in_box_mask,
                                  pad_width=((padding // 2, padding // 2), (padding // 2, padding // 2)),
                                  mode="constant")  # (min_side, min_side)
        proj_in_box_mask = proj_in_box_mask.astype(float) / 255.0  # (min_side, min_side)
        proj_in_box_mask = proj_in_box_mask[..., np.newaxis]  # (min_side, min_side, 1)
    else:
        stop_sign_img = traffic_signs_cut[0]
        oct_mask = stop_sign_img[..., 3][..., np.newaxis]  # (h, w, 1)
        proj_in_box_mask = cv2.resize((oct_mask*255).astype(np.uint8),
                                      (cfg.grid_size - padding, cfg.grid_size - padding), interpolation=cv2.INTER_CUBIC)
        proj_in_box_mask[proj_in_box_mask > 0.0] = 255
        proj_in_box_mask = np.pad(proj_in_box_mask,
                                  pad_width=((padding // 2, padding // 2), (padding // 2, padding // 2)),
                                  mode="constant")  # (min_side, min_side)
        proj_in_box_mask = proj_in_box_mask.astype(float) / 255.0  # (min_side, min_side)
        proj_in_box_mask = proj_in_box_mask[..., np.newaxis]  # (min_side, min_side, 1)

    # perspective transform
    angles_h = np.clip(np.abs(np.random.normal(0, 15, size=samples_no)), 0, cfg.max_angle_h)
    ht = [get_horizontal_perspective_transformer(heights[i], angles_h[i]) for i in range(samples_no)]
    angles_v = np.clip(np.abs(np.random.normal(0, 5, size=samples_no)), 0, cfg.max_angle_v)
    vt = [get_vertical_perspective_transformer(heights[i], angles_v[i]) for i in range(samples_no)]
    lor = np.round(np.random.rand(samples_no))
    uod = np.round(np.random.rand(samples_no))

    # rotations
    rotation_angles = np.random.uniform(low=-cfg.rotation_angle_abs, high=cfg.rotation_angle_abs, size=(samples_no))

    # brightness pre
    brightnesses_pre = []
    for i in range(cfg.batches_before_checkpoint):
        for j in range(cfg.batch_size):
            brightnesses_pre.append((np.random.rand()*2-1) * cfg.brightness_variation_pre)
    brightnesses_pre = np.array(brightnesses_pre)

    # brightness post
    brightnesses_post= []
    for i in range(cfg.batches_before_checkpoint):
        for j in range(cfg.batch_size):
            brightnesses_post.append((np.random.rand()*2-1) * cfg.brightness_variation_post)
    brightnesses_post = np.array(brightnesses_post)

    # gaussian blur
    gaussian_blur_stds = cfg.gauss_blur_filter_std_min + np.random.rand(samples_no)*(cfg.gauss_blur_filter_std_max - cfg.gauss_blur_filter_std_min)

    # channel specific offset
    channel_offset = np.random.rand(samples_no, 3) * cfg.channel_offset

    # traffic sign box pad
    traffic_sign_box_pad = cfg.input_pad_min + np.random.rand(samples_no, 2)*cfg.input_pad_max

    generated["batch_bg_imgs"] = batch_bg_imgs
    generated["batch_ts_imgs"] = batch_ts_imgs
    generated["boxes"] = boxes
    generated["shifts"] = shifts
    generated["proj_in_box_mask"] = proj_in_box_mask
    generated["perspective_transform_h"] = ht
    generated["perspective_transform_v"] = vt
    generated["perspective_left_or_right"] = lor
    generated["perspective_up_or_down"] = uod
    generated["rotation_angles"] = rotation_angles
    generated["rotation_pad"] = np.ones(samples_no).astype(int) * cfg.rotation_pad
    generated["brightnesses_pre"] = brightnesses_pre
    generated["brightnesses_post"] = brightnesses_post
    generated["gaussian_blur_std"] = gaussian_blur_stds
    generated["gaussian_blur_fs"] = np.ones(samples_no).astype(int) * cfg.gauss_blur_filter_size
    generated["channel_offset"] = channel_offset
    generated["traffic_sign_box_pad"] = traffic_sign_box_pad

    return generated
