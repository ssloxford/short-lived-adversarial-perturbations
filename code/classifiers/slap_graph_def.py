import tensorflow as tf
import cv2
import numpy as np


def make_loss_function(slappy, scores, class_index, batch_size, pnorm_mult, tv_mult, target_mult, target_class=None):
    # sum of all scores that are a stop sign
    det_scores_sum = tf.reduce_sum(scores[..., class_index]**2)
    if target_class is not None:
        det_scores_sum = det_scores_sum + tf.reduce_sum((1-scores[..., target_class])**2*target_mult)
    # define loss function
    total_variation = tf.reduce_sum(tf.image.total_variation(slappy["projection_tanh"]))
    # regularize some of these with weights
    total_variation_reg = tf.pow(total_variation / 1000.0, 1) * tv_mult
    # perceptibility
    pnorm = []
    for i in range(batch_size):
        _norm = tf.norm(slappy["proj_model_deltas"][i], ord=np.inf)
        pnorm.append(_norm)
    pnorm = tf.reduce_mean(tf.stack(pnorm, axis=0))*pnorm_mult/batch_size
    loss_function = total_variation_reg + det_scores_sum + pnorm  # + nps_reg)
    return loss_function, total_variation_reg, det_scores_sum, pnorm


def interp_enum(method, target):
    assert method in ["linear", "nearest", "area", "cubic"]
    assert target in ["cv2", "tf"]
    if target == "cv2":
        if method == "linear":
            return cv2.INTER_LINEAR
        if method == "nearest":
            return cv2.INTER_NEAREST
        if method == "area":
            return cv2.INTER_AREA
        if method == "cubic":
            return cv2.INTER_CUBIC
    if target == "tf":
        if method == "linear":
            return tf.image.ResizeMethod.BILINEAR
        if method == "nearest":
            return tf.image.ResizeMethod.NEAREST_NEIGHBOR
        if method == "area":
            return tf.image.ResizeMethod.AREA
        if method == "cubic":
            return tf.image.ResizeMethod.BICUBIC
    return -1


def slap_initialize_vars(sess, proj_model_dict, slappy):
    adam_vars_to_init = get_adam_vars()
    proj_model_assign_ops = get_proj_model_assign_ops(proj_model_dict)

    init_vars = tf.compat.v1.variables_initializer([slappy["projection_base"]] + adam_vars_to_init)

    sess.run([init_vars, proj_model_assign_ops])
    return sess


def get_adam_vars():
    l = []
    for var in tf.compat.v1.global_variables():
        if "adam/" in var.name:
            l.append(var)
    return l


def get_proj_model_vars():
    v = []
    for var in tf.compat.v1.global_variables():
        if "proj_model/" in var.name:
            v.append(var)
    return v


def get_proj_model_assign_ops(proj_model_dict):
    proj_model_vars_to_init = get_proj_model_vars()
    proj_model_assign_ops = []
    for var in proj_model_vars_to_init:
        # var names look like this "slap/proj_model/dense_layer1_b1/bias:0"
        _, _, layerid, b_or_k = var.name.split("/")
        layerid = "_".join(layerid.split("_")[:-1])
        b_or_k = b_or_k[:-2]
        op = var.assign(proj_model_dict["proj_model/{}/{}:0".format(layerid, b_or_k)])
        proj_model_assign_ops.append(op)
    return proj_model_assign_ops


def apply_gaussian_blur(image, filter_size, filter_std):
    gauss_kernel = gaussian_kernel(filter_size, 0.0, filter_std)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    gauss_kernel = tf.concat((gauss_kernel, gauss_kernel, gauss_kernel), axis=-2)
    img_exp_dim = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(img_exp_dim, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    return blurred[0]


def add_projection_model(origins, projections, box_coordinates, ss_inbox_mask, batch_size, pm_l1, pibm, bypass=False):
    """

    :param origins: list of tensors with the input stop signs
    :param projections: list of projections
    :param batch_size:
    :return:
    """
    deltas, outputs, patches = [], [], []
    with tf.compat.v1.variable_scope("proj_model"):
        if not bypass:
            for i in range(batch_size):
                proj_model_input_org = tf.reshape(origins[i], [-1, 3])  # (bh * bw, 3)
                proj_model_input_pert = tf.reshape(projections[i], [-1, 3])  # (bh * bw, 3)

                proj_model_input = tf.concat((proj_model_input_org, proj_model_input_pert), axis=1)
                l1 = tf.keras.layers.Dense(
                    pm_l1, activation=tf.nn.relu, name="dense_layer1_b%d" % i, trainable=False)(proj_model_input)
                l2 = tf.keras.layers.Dense(
                    3, activation=tf.nn.relu, name="dense_layer2_b%d" % i, trainable=False)(l1)
                proj_model_out = tf.keras.activations.tanh(l2)
                proj_model_out = tf.reshape(
                    proj_model_out,
                    [box_coordinates[i, 3] - box_coordinates[i, 1], box_coordinates[i, 2] - box_coordinates[i, 0],
                     3])  # (bh, bw, 3)
                # avoid transparent part effects
                proj_model_out = proj_model_out * ss_inbox_mask[i]  # (bh, bw, 3) x (bh, bw, 1)
                proj_model_out = tf.identity(proj_model_out, name="proj_model_output_%d" % i)
                patch = tf.identity(proj_model_out, name="patch_%d" % i)

                proj_model_delta = tf.abs(tf.reshape(proj_model_out, [-1, 3]) - proj_model_input_org)
                proj_model_delta = tf.reshape(proj_model_delta, [box_coordinates[i, 3] - box_coordinates[i, 1],
                                                                 box_coordinates[i, 2] - box_coordinates[i, 0], 3])
                proj_model_delta = proj_model_delta * ss_inbox_mask[i]
                proj_model_delta = tf.identity(proj_model_delta, name="proj_model_delta_%d" % i)

                deltas.append(proj_model_delta)
                outputs.append(proj_model_out)
                patches.append(patch)
        else:
            for i in range(batch_size):
                _tmp = projections[i] * pibm[i]
                origins_with_hole = origins[i] * (1-pibm[i])

                proj_model_out = tf.clip_by_value((origins_with_hole+_tmp)*ss_inbox_mask[i], 0.0, 1.0)
                proj_model_out = tf.identity(proj_model_out, name="proj_model_output_%d" % i)

                patch = tf.identity(proj_model_out * pibm[i], name="patch_%d" % i)

                proj_model_delta = tf.clip_by_value(tf.abs(proj_model_out - origins[i]), 0, 1)
                proj_model_delta = tf.identity(proj_model_delta, name="proj_model_delta_%d" % i)
                deltas.append(proj_model_delta)
                outputs.append(proj_model_out)
                patches.append(patch)

    return deltas, outputs, patches


def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def slap_angle_d_to_persp_transform(imgs, img_masks, transformer_h, transformer_v, uod, lor, batch_size):
    imgs_transformed, masks_transformed = [], []
    for i in range(batch_size):
        img_flipped = tf.cond(lor[i]>0, lambda: tf.image.flip_left_right(imgs[i]), lambda: imgs[i])
        mask_flipped = tf.cond(lor[i]>0, lambda: tf.image.flip_left_right(img_masks[i]), lambda: img_masks[i])
        img_transf = tf.contrib.image.transform(img_flipped, transformer_h[i], interpolation='BILINEAR')
        mask_transf = tf.contrib.image.transform(mask_flipped, transformer_h[i], interpolation='BILINEAR')
        img_h = tf.cond(lor[i]>0, lambda: tf.image.flip_left_right(img_transf), lambda: img_transf)
        mask_h = tf.cond(lor[i]>0, lambda: tf.image.flip_left_right(mask_transf), lambda: mask_transf)
        img_h = tf.identity(img_h, name="stop_sign_persp_transf_h_b%d" % i)
        mask_h = tf.identity(mask_h, name="stop_sign_mask_persp_transf_h_b%d" % i)

        img_flipped2 = tf.cond(uod[i]>0, lambda: tf.image.flip_up_down(img_h), lambda: img_h)
        mask_flipped2 = tf.cond(uod[i]>0, lambda: tf.image.flip_up_down(mask_h), lambda: mask_h)
        img_transf2 = tf.contrib.image.transform(img_flipped2, transformer_v[i], interpolation='BILINEAR')
        mask_transf2 = tf.contrib.image.transform(mask_flipped2, transformer_v[i], interpolation='BILINEAR')
        img_v = tf.cond(uod[i]>0, lambda: tf.image.flip_up_down(img_transf2), lambda: img_transf2)
        mask_v = tf.cond(uod[i]>0, lambda: tf.image.flip_up_down(mask_transf2), lambda: mask_transf2)
        img_v = tf.identity(img_v, name="stop_sign_persp_transf_v_b%d" % i)
        mask_v = tf.identity(mask_v, name="stop_sign_mask_persp_transf_v_b%d" % i)

        imgs_transformed.append(img_v)
        masks_transformed.append(mask_v)
    return imgs_transformed, masks_transformed


def add_slap_instructions(res_x, res_y, batch_size, pm_l1, resize_method, init, bypass_pm=False, trainable=True):

    return_tensors = {}

    with tf.compat.v1.variable_scope('slap'):
        # override yolo placeholder
        input_bg_imgs = tf.compat.v1.placeholder(tf.float32, [None, 416, 416, 3], name='input_bg_imgs')
        traffic_sign_imgs = tf.compat.v1.placeholder(tf.float32, [None, 256, 256, 4], name='traffic_sign_imgs')
        box_coords = tf.compat.v1.placeholder(tf.int32, [None, 4], name='box_coordinates')
        projection_in_box_mask = tf.compat.v1.placeholder(tf.float32, [res_x, res_y, 1], name='projection_in_box_mask')
        projection_shift = tf.compat.v1.placeholder(tf.int32, [None, 2], name='projection_shift')
        perspective_transform_h = tf.compat.v1.placeholder(tf.float32, [None, 8], name='perspective_transform_h')
        perspective_transform_v = tf.compat.v1.placeholder(tf.float32, [None, 8], name='perspective_transform_v')
        perspective_up_or_down = tf.compat.v1.placeholder(tf.float32, [None], name='perspective_up_or_down')
        perspective_left_or_right = tf.compat.v1.placeholder(tf.float32, [None], name='perspective_left_or_right')
        brightness_variation_pre = tf.compat.v1.placeholder(tf.float32, [None], name='brightness_variation_pre')
        brightness_variation_post = tf.compat.v1.placeholder(tf.float32, [None], name='brightness_variation_post')
        rotation_angle = tf.compat.v1.placeholder(tf.float32, [None], name='rotation_angle')
        rotation_pad = tf.compat.v1.placeholder(tf.float32, [None], name='rotation_pad')
        gauss_blur_std = tf.compat.v1.placeholder(tf.float32, [None], name='gauss_blur_std')
        gauss_blur_fs = tf.compat.v1.placeholder(tf.float32, [None], name='gauss_blur_fs')
        channel_offset = tf.compat.v1.placeholder(tf.float32, [None, 3], name='channel_offset')
        traffic_sign_box_pad = tf.compat.v1.placeholder(tf.float32, [None, 2], name='traffic_sign_box_pad')

        initial_value = {
            "zeros": tf.zeros(shape=[res_x, res_y, 3]),
            "ones": tf.ones(shape=[res_x, res_y, 3]),
            "rand": tf.random.uniform(shape=[res_x, res_y, 3]),
            "min_ones": tf.ones(shape=[res_x, res_y, 3])*-1}

        # perturbation variable
        projection_base = tf.Variable(
            dtype=tf.float32,
            initial_value=initial_value[init],
            trainable=trainable,
            name="projection_base",
        )

        projection_tanh = tf.keras.activations.tanh(projection_base)  # bounded to [-1, 1]
        projection_tanh = projection_tanh/2+0.5  # bounded to [0, 1]

        # masked projection
        masked_projection = projection_in_box_mask * projection_tanh   # (res_x, res_y, 3)
        masked_projection = tf.identity(masked_projection, name="masked_projection")  # (res_x, res_y, 3)

        # 1. resize: (i) stop sign, (ii) stop sign in box mask, (iii) projection, (iv) projection in box mask
        # 2. apply input shift to projection and its mask
        projs_r_s, projs_in_box_mask_r_s, stop_sign_imgs_r, stop_sign_in_box_masks = [], [], [], []
        for i in range(batch_size):
            box_height, box_width = box_coords[i, 3] - box_coords[i, 1], box_coords[i, 2] - box_coords[i, 0]
            proj_r = tf.image.resize(masked_projection, (box_height, box_width),
                                                  method=interp_enum(resize_method, "tf"))  # (bh, bw, 3)
            proj_in_box_mask_r = tf.image.resize(projection_in_box_mask, (box_height, box_width),
                                                  method=interp_enum("nearest", "tf"))  # (bh, bw, 3)
            # drop alpha channel
            stop_sign_img_r = tf.image.resize(traffic_sign_imgs[i][..., :3], (box_height, box_width),
                                                  method=interp_enum("linear", "tf"))  # (bh, bw, 3)

            stop_sign_in_box_mask = tf.image.resize(
                tf.expand_dims(traffic_sign_imgs[i][..., 3], axis=-1), (box_height, box_width),
                method=interp_enum("nearest", "tf"))  # (bh, bw, 1)

            stop_sign_imgs_r.append(stop_sign_img_r)
            stop_sign_in_box_masks.append(stop_sign_in_box_mask)

            # shift and name the projection
            projection_r_s = tf.roll(proj_r, projection_shift[i, 0], axis=0)  # (bh, bw, 3)
            projection_r_s = tf.roll(projection_r_s, projection_shift[i, 1], axis=1)
            projection_r_s = tf.identity(projection_r_s, name="projection_resized_shifted_%d" % i)

            # shift and name the projection mask for future use
            proj_in_box_mask_r_s = tf.roll(proj_in_box_mask_r, projection_shift[i, 0], axis=0)   # (bh, bw, 3)
            proj_in_box_mask_r_s = tf.roll(proj_in_box_mask_r_s, projection_shift[i, 1], axis=1)
            proj_in_box_mask_r_s = tf.round(proj_in_box_mask_r_s)  # makes values either 0 or 1
            proj_in_box_mask_r_s = tf.identity(proj_in_box_mask_r_s, name="mask_resized_shifted_%d" % i)

            projs_in_box_mask_r_s.append(proj_in_box_mask_r_s)
            projs_r_s.append(projection_r_s)

        # 3. apply brightness variation to stop sign images pre
        stop_sign_imgs_r_bv = []
        for i in range(batch_size):
            stop_sign_img_r_bv = tf.image.adjust_brightness(stop_sign_imgs_r[i], delta=brightness_variation_pre[i])  # (bh, bw, 3)
            stop_sign_img_r_bv = tf.clip_by_value(stop_sign_img_r_bv * stop_sign_in_box_masks[i], 0.0, 1.0)
            stop_sign_img_r_bv = tf.identity(stop_sign_img_r_bv, name="stop_sign_img_r_bv_pre_%d" % i)
            stop_sign_imgs_r_bv.append(stop_sign_img_r_bv)

        # 4. apply projection model
        proj_model_deltas, proj_model_outs, patches = add_projection_model(stop_sign_imgs_r_bv, projs_r_s, box_coords,
            stop_sign_in_box_masks, batch_size, pm_l1, projs_in_box_mask_r_s, bypass=bypass_pm)

        # 4b. apply brightness variation post projection model
        proj_model_outs_bv = []
        for i in range(batch_size):
            proj_model_out_bv = tf.image.adjust_brightness(proj_model_outs[i],
                                                            delta=brightness_variation_post[i])  # (bh, bw, 3)
            proj_model_out_bv = tf.clip_by_value(proj_model_out_bv * stop_sign_in_box_masks[i], 0.0, 1.0)
            proj_model_out_bv = tf.identity(proj_model_out_bv, name="stop_sign_img_r_bv_post_%d" % i)
            proj_model_outs_bv.append(proj_model_out_bv)

        # 5. add perspective transform
        stop_signs_pts, ss_in_box_masks_pts = slap_angle_d_to_persp_transform(proj_model_outs_bv, stop_sign_in_box_masks,
            perspective_transform_h, perspective_transform_v, perspective_up_or_down, perspective_left_or_right, batch_size)

        stop_signs_noised = []
        ss_in_box_masks_noised = []
        for i in range(batch_size):
            red_noise = tf.random.uniform([1], minval=-channel_offset[i][0], maxval=channel_offset[i][0])[0]
            green_noise = tf.random.uniform([1], minval=-channel_offset[i][0], maxval=channel_offset[i][0])[0]
            blue_noise = tf.random.uniform([1], minval=-channel_offset[i][0], maxval=channel_offset[i][0])[0]
            noise = tf.stack([red_noise, green_noise, blue_noise], axis=0)
            o_shape = tf.shape(stop_signs_pts[i])
            _with_noise = tf.reshape(stop_signs_pts[i], (-1, 3))
            _with_noise = _with_noise + noise
            _with_noise = tf.reshape(_with_noise, o_shape)
            _mask = ss_in_box_masks_pts[i]
            _with_noise = tf.clip_by_value(_with_noise, 0.0, 1.0)
            stop_signs_noised.append(_with_noise)
            ss_in_box_masks_noised.append(_mask)

        # 6. rotation (plus padding to avoid corner cuts)
        stop_signs_rots = []
        ss_in_box_masks_rots = []
        for i in range(batch_size):
            paddings = [[rotation_pad[i], rotation_pad[i]], [rotation_pad[i], rotation_pad[i]], [0, 0]]
            ss_pad = tf.pad(stop_signs_noised[i], paddings, "CONSTANT")
            ss_mask_pad = tf.pad(ss_in_box_masks_noised[i], paddings, "CONSTANT")
            ss_rot = tf.contrib.image.transform(ss_pad,
                tf.contrib.image.angles_to_projective_transforms(
                    rotation_angle[i], tf.cast(tf.shape(ss_pad)[0], tf.float32), tf.cast(tf.shape(ss_pad)[1], tf.float32)))
            ss_mask_rot = tf.contrib.image.transform(ss_mask_pad,
                tf.contrib.image.angles_to_projective_transforms(
                    rotation_angle[i], tf.cast(tf.shape(ss_mask_pad)[0], tf.float32), tf.cast(tf.shape(ss_mask_pad)[1], tf.float32)))
            stop_signs_rots.append(ss_rot)
            ss_in_box_masks_rots.append(ss_mask_rot)

        # 7. final pad and generate final image
        final_imgs = []
        for i in range(batch_size):
            box_height, box_width = box_coords[i, 3] - box_coords[i, 1], box_coords[i, 2] - box_coords[i, 0]

            pole_center_x = tf.math.floordiv(box_coords[i, 2] + box_coords[i, 0], 2)  #  - tf.math.floordiv(box_width, 10)
            pole_center_y = tf.math.floordiv(box_coords[i, 3] + box_coords[i, 1], 2)

            pole_height = tf.cast(tf.cast(box_height, dtype=tf.float32) * 1.5, tf.int32)
            pole_height = tf.minimum(416 - pole_center_y, pole_height)
            pole_width = tf.cast(tf.maximum(1.0, tf.cast(box_height, dtype=tf.float32) * 0.075), tf.int32)

            pole_mask = tf.ones((pole_height, pole_width, 3), dtype=tf.float32)

            pole_x1 = pole_center_x - tf.math.floordiv(pole_width, 2)
            pole_x1 = tf.identity(pole_x1, name="pole_x1")

            pole_y1 = pole_center_y
            pole_y1 = tf.identity(pole_y1, name="pole_y1")

            pole_gradient = tf.linspace(0.4, 0.6, pole_width, name="linspace")  # (pole_width, )
            pole_gradient = tf.expand_dims(tf.expand_dims(pole_gradient, axis=-1), axis=0)  # (1, pole_width, 1)
            pole = pole_mask * pole_gradient  # (ph, pw, 3)

            pole_mask_padded = tf.image.pad_to_bounding_box(pole_mask, pole_y1, pole_x1, 416, 416)  # (416, 416, 3)
            pole_padded = tf.image.pad_to_bounding_box(pole, pole_y1, pole_x1, 416, 416)  # (416, 416, 3)
            bg_with_pole_hole = (1 - pole_mask_padded) * input_bg_imgs[i]
            bg_with_pole = bg_with_pole_hole + (pole_mask_padded * pole_padded)

            # 6a pad stop sign
            padded_ss = tf.image.pad_to_bounding_box(stop_signs_rots[i], box_coords[i, 1]-tf.cast(rotation_pad[i], tf.int32),
                                                     box_coords[i, 0]-tf.cast(rotation_pad[i], tf.int32), 416, 416)  # (416, 416, 3)
            # 6b pad stop sign in box mask
            padded_mask = tf.image.pad_to_bounding_box(ss_in_box_masks_rots[i], box_coords[i, 1]-tf.cast(rotation_pad[i], tf.int32),
                                                       box_coords[i, 0]-tf.cast(rotation_pad[i], tf.int32), 416, 416)  # (416, 416, 3)
            # 6c flip padded mask 0>1 and 1>0
            padded_mask = (padded_mask-1)*-1  # (416, 416, 3)
            bg_with_ss_hole = bg_with_pole * padded_mask  # (416, 416, 3)
            bg_with_sign = bg_with_ss_hole + padded_ss  # (416, 416, 3)
            bg_with_sign = tf.clip_by_value(bg_with_sign, 0.0, 1.0)
            bg_with_blur = apply_gaussian_blur(bg_with_sign, gauss_blur_fs[i], gauss_blur_std[i])
            final_imgs.append(tf.expand_dims(bg_with_blur, axis=0))

        final_imgs = tf.concat(final_imgs, axis=0)  # (b, 416, 416, 3)
        final_imgs = tf.identity(final_imgs, name="slapped_input")  # (b, 416, 416, 3)

        org_widths = tf.cast(box_coords[:, 2] - box_coords[:, 0], tf.float32)
        org_heights = tf.cast(box_coords[:, 3] - box_coords[:, 1], tf.float32)
        pad_width = tf.maximum(0, tf.cast(org_widths * traffic_sign_box_pad[:, 0], tf.int32))
        pad_height = tf.maximum(0, tf.cast(org_heights * traffic_sign_box_pad[:, 1], tf.int32))

        pad_box_x1 = tf.maximum(0, box_coords[:, 0] - pad_width)
        pad_box_y1 = tf.maximum(0, box_coords[:, 1] - pad_height)
        pad_box_x2 = tf.minimum(box_coords[:, 2] + pad_width, 416)
        pad_box_y2 = tf.minimum(box_coords[:, 3] + pad_height, 416)

        box_coords_padded = tf.transpose(tf.stack((pad_box_x1, pad_box_y1, pad_box_x2, pad_box_y2)))
        box_coords_padded = tf.identity(box_coords_padded, name="box_coords_padded")

    # images to start from
    return_tensors["input_bg_imgs"] = input_bg_imgs
    return_tensors["traffic_sign_imgs"] = traffic_sign_imgs
    return_tensors["box_coords"] = box_coords
    return_tensors["box_coords_padded"] = box_coords_padded
    return_tensors["projection_in_box_mask"] = projection_in_box_mask
    return_tensors["projection_base"] = projection_base
    return_tensors["projection_tanh"] = projection_tanh
    return_tensors["masked_projection"] = masked_projection

    # things that add variability
    return_tensors["projection_shift"] = projection_shift
    return_tensors["brightness_variation_pre"] = brightness_variation_pre
    return_tensors["brightness_variation_post"] = brightness_variation_post
    return_tensors["perspective_transform_h"] = perspective_transform_h
    return_tensors["perspective_transform_v"] = perspective_transform_v
    return_tensors["perspective_left_or_right"] = perspective_left_or_right
    return_tensors["perspective_up_or_down"] = perspective_up_or_down
    return_tensors["rotation_pad"] = rotation_pad
    return_tensors["rotation_angle"] = rotation_angle
    return_tensors["gauss_blur_std"] = gauss_blur_std
    return_tensors["gauss_blur_fs"] = gauss_blur_fs
    return_tensors["channel_offset"] = channel_offset
    return_tensors["traffic_sign_box_pad"] = traffic_sign_box_pad

    # things in output
    # 1./2. resize and shift
    return_tensors["projs_r_s"] = projs_r_s
    return_tensors["projs_in_box_mask_r_s"] = projs_in_box_mask_r_s
    return_tensors["stop_sign_imgs_r"] = stop_sign_imgs_r
    return_tensors["stop_sign_in_box_masks"] = stop_sign_in_box_masks
    # 3. brightness
    return_tensors["stop_sign_imgs_r_bv"] = stop_sign_imgs_r_bv
    # 4. projection model
    return_tensors["proj_model_outs"] = proj_model_outs
    # 4a. patches if necessary
    return_tensors["patches"] = patches
    # 4b. projection model absolute delta to compute l-p norm
    return_tensors["proj_model_deltas"] = proj_model_deltas
    # 5. perspective transform
    return_tensors["stop_signs_pts"] = stop_signs_pts
    # 6. rotate
    # return_tensors["stop_signs_rots"] = stop_signs_rots
    # 7. final pad and overlay
    return_tensors["slapped_input"] = final_imgs

    return return_tensors
