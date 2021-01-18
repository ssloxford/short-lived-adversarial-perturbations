import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import cv2


def get_moments(cont):
    M = cv2.moments(cont)
    cX, cY = 0, 0
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return cX, cY


def get_exposure_abs():
    check_exposure_abs_cmd = ["v4l2-ctl", "--get-ctrl=exposure_absolute"]
    output_abs_exp = subprocess.check_output(check_exposure_abs_cmd).decode("utf-8")
    output_abs_exp = output_abs_exp.rstrip("\n").split(":")[-1].strip(" ")
    exposure_v = int(output_abs_exp)
    return exposure_v


def set_exposure_abs(value):
    assert 0<value<2048
    print("Exposure set to %s" %value)
    set_exposure_val_cmd = "v4l2-ctl --set-ctrl=exposure_absolute=%d"
    subprocess.call((set_exposure_val_cmd % value).split(" "))
    exposure_v = get_exposure_abs()
    return exposure_v


def get_exposure_auto():
    check_auto_exposure_cmd = [
        "v4l2-ctl",
        "--get-ctrl=exposure_auto"
    ]
    output_auto_exp = subprocess.check_output(check_auto_exposure_cmd).decode("utf-8")
    output_auto_exp = output_auto_exp.rstrip("\n").split(":")[-1].strip(" ")
    return output_auto_exp


def set_exposure_auto(value):
    assert value in [1, 3]
    set_auto_exposure_cmd = "v4l2-ctl --set-ctrl=exposure_auto=%d"
    subprocess.call((set_auto_exposure_cmd % 1).split(" "))
    exp_auto_v = get_exposure_auto()
    return exp_auto_v


def find_exposure(webcam, window_name="Capturing", window_loc=(20, 20)):

    try:
        # output_auto_exp = subprocess.check_output(check_auto_exposure_cmd).decode("utf-8")
        # output_auto_exp = output_auto_exp.rstrip("\n").split(":")[-1].strip(" ")

        v = set_exposure_auto(1)
        assert v == "1", "Exposure auto returned %s" % v
        exposure_v = get_exposure_abs()
        exposure_initial = exposure_v

        while True:
            check, frame = webcam.read()

            frame = cv2.putText(frame, "A: reduce exposure", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), lineType=cv2.LINE_AA)
            frame = cv2.putText(frame, "D: increase exposure", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), lineType=cv2.LINE_AA)
            frame = cv2.putText(frame, "Exposure=%d (from %d)" % (exposure_v, exposure_initial), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0),lineType=cv2.LINE_AA)
            frame = cv2.putText(frame, "C: confirm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0),lineType=cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            cv2.moveWindow(window_name, window_loc[0], window_loc[0])
            key = cv2.waitKey(1)

            if key == ord("d"):
                exposure_v = set_exposure_abs(min(2000, exposure_v + 10))

            if key == ord("a"):
                exposure_v = set_exposure_abs(max(1, exposure_v - 10))

            if key == ord('c'):
                print('Final exposure is %s' % exposure_v)
                cv2.destroyAllWindows()
                break

    except Exception as e:
        print("Failed miserably", e)
        return False
    return exposure_v


def get_dict_for_proj(fpath):
    df = pd.read_csv(fpath, index_col=False)
    df = df.drop(columns="outcome_std")
    p_org = df.values[:, 0:3]

    # I do not take responsibility for this
    p_org_uniq = list(sorted(list(set(list(map(tuple, p_org.tolist()))))))

    m = 0
    for r, g, b in p_org_uniq:
        m = np.maximum(m, df[(df["origin_r"]==r) & (df["origin_g"]==g) & (df["origin_b"]==b)].shape[0])
    #dct = {}
    big_arr_prj = np.zeros(shape=(0, m, 6), dtype=int)
    for r, g, b in p_org_uniq:
        prj = df[(df["origin_r"]==r) & (df["origin_g"]==g) & (df["origin_b"]==b)].values[:, 6:9]
        add = df[(df["origin_r"]==r) & (df["origin_g"]==g) & (df["origin_b"]==b)].values[:, 3:6]
        if prj.shape[0] < m:
            prj = np.pad(prj, ((0, np.abs(prj.shape[0]-m)), (0, 0)), mode="edge")
            add = np.pad(add, ((0, np.abs(add.shape[0]-m)), (0, 0)), mode="edge")
        #dct[(r, g, b)] = prj

        big_arr_prj = np.concatenate((big_arr_prj, np.hstack((prj, add))[np.newaxis, :]))

    # add a black origin to black projection so that it is ignored in nps computation
    big_arr_prj = np.vstack((np.zeros(shape=(1, m, 6)), big_arr_prj))
    p_org_uniq = np.vstack((np.zeros(3, ), p_org_uniq))
    return big_arr_prj.astype(float)/255.0, p_org_uniq.astype(float)/255.0


def plot_projectables_img(matrix, unique_org, fname="projectables.jpg"):
    """

    :param matrix: projectable matrix, shape (n, m, 6) where n is the # of origin pixels, m the # of projectables per origin
     with elements 0:3 being the projectable and 3:6 being the additional colour
    :param unique_org: (n, 3) unique origin pixels
    :param fname: filename to save the image into
    :return:
    """
    n_org = matrix.shape[0]
    n_cols = 5
    n_rows = n_org//n_cols + 1
    print(n_org, n_rows)
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, n_org), )  #gridspec_kw={'width_ratios': [1, 20]})
    for origin_i in range(n_cols*n_rows):
        row_i = origin_i // n_cols
        col_i = origin_i % n_cols
        ax[row_i, col_i].set_xticks([])
        ax[row_i, col_i].set_yticks([])
        ax[row_i, col_i].imshow([[[0, 0, 0]]])
        if origin_i < n_org:

            ax[row_i, col_i].set_title("%s" % (unique_org[origin_i] * 255).astype(int), fontsize=30)

            # plt.figure(figsize=(16, 3))
            prj_stack = matrix[origin_i][:, 0:3][np.newaxis]
            add_stack = matrix[origin_i][:, 3:6][np.newaxis]
            prj_stack = np.concatenate((prj_stack, prj_stack, prj_stack, prj_stack, prj_stack), axis=0)
            add_stack = np.concatenate((add_stack, add_stack, add_stack, add_stack, add_stack), axis=0)
            stack = np.concatenate((add_stack, prj_stack), axis=0)

            org_img = unique_org[origin_i][np.newaxis][np.newaxis]  # (1, 1, 3)
            prj_img = stack  # (5+5, nadd, 3)
            org_img = np.concatenate((org_img, org_img, org_img, org_img, org_img, org_img, org_img, org_img, org_img, org_img, ), axis=1)
            org_img = np.concatenate((org_img, org_img, org_img, org_img, org_img, org_img, org_img, org_img, org_img, org_img, ), axis=0)  # (10, 10, 3)
            to_show = np.concatenate((org_img, prj_img), axis=1)
            ax[row_i, col_i].imshow(to_show)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def origins_histogram(fpath, output_fname):
    df = pd.read_csv(fpath, index_col=False)
    p_org = df.values[:, 0:3]

    # I do not take responsibility for this
    p_org_uniq = list(sorted(list(set(list(map(tuple, p_org.tolist()))))))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sizes = {}
    for i in range(len(p_org_uniq)):
        # count how many are like this one
        this_org = p_org_uniq[i]
        this_org_counter = 1
        for j in range(len(p_org)):
            if tuple(p_org[j]) == tuple(this_org):
                this_org_counter += 5
        sizes[this_org] = this_org_counter
    max_size = np.array([sizes[i] for i in p_org_uniq]).max()
    # min_size = np.array([sizes[j] j in range(len(p_org)]).min()
    maximum_size_point = 1000
    size_factor = maximum_size_point/max_size
    for i in range(len(p_org_uniq)):
        to = p_org_uniq[i]
        ax.scatter(to[0], to[1], to[2], c=np.array((to[0]/255, to[1]/255, to[2]/255))[np.newaxis, ...], s=sizes[to]*size_factor)
        ax.text(to[0], to[1], to[2],  '%s' % (str(sizes[to])), size=5, zorder=100, color='k') 
    ax.set_xlabel('r')
    ax.set_ylabel('g')
    ax.set_zlabel('b')
    ax.set_xlim3d(0, 200)
    ax.set_ylim3d(0, 200)
    ax.set_zlim3d(0, 200)
    plt.tight_layout()
    plt.savefig(output_fname, bbox_inches="tight")
    plt.close()
