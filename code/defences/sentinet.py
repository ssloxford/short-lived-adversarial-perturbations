import argparse
import numpy as np
import os
import pandas as pd
import yaml
from tqdm import tqdm
from scipy import optimize
from classifiers.lisacnn.detector import LisaCNNModel
from classifiers.gtsrbcnn.detector import GtsrbCNNModel


def quadratic_func(x, a, b, c):
    return a*x**2 + b*x + c


class Sentinet(object):
    def __init__(self, model):
        self.model = model

    # Algorithm 1
    def class_proposal(self, img, num_classes=5):
        # We perform our class proposal by using the top 5 classes for our given sign (excluding stop) + stop sign
        label_probabilities = self.model.forward(img, save_image=False, probs_only=True)
        top_classes = list(label_probabilities.argsort()[-(num_classes + 1):][::-1])
        prediction = (top_classes[0], label_probabilities[top_classes][0])
        final_set = [(x, label_probabilities[x]) for x in top_classes[1:]]
        return prediction, final_set

    # Algorithm 2
    def mask_generation(self, img, prediction, proposed_classes, threshold=0.3, saliency_function=None):

        if saliency_function is None:
            saliency_function = self.model.xrai

        mask_y = saliency_function(img, prediction[0], binarize=True, threshold=threshold)
        mask_set = []
        for (yp_class, yp_conf) in proposed_classes:
            mask_yp = saliency_function(img, yp_class, binarize=True, threshold=threshold)
            delta_mask = mask_y & ~mask_yp
            mask_set.append({'mask': delta_mask, 'confidence': yp_conf, 'class': yp_class})
        return mask_set, mask_y

    # Algorithm 3
    def testing(self, img, adv_example_class, masks, test_image_paths, pattern='noise'):

        R = []
        IP = []
        if pattern == 'noise':
            inert_pattern = (np.random.random(img.shape) * 255).astype(np.uint8)
        elif pattern == 'checker':
            inert_pattern = self.model.load_image('/home/code/images/pattern.png')
        else:
            raise ValueError('Unsupported pattern, choose one of %s' % ["noise", "checker"])

        for m in masks:
            img_mask = np.copy(img)
            img_mask[~m['mask']] = 0
            R.append(img_mask)
            inert_mask = np.copy(inert_pattern)
            inert_mask[~m['mask']] = 0
            IP.append(inert_mask)

        Xr = []
        Xip = []
        for test_im_path in test_image_paths:
            image = self.model.load_image(test_im_path)
            for r in R:
                mask = (r.sum(axis=-1) == 0)[..., np.newaxis].astype(np.uint8)
                new_image = r + image*mask
                Xr.append(new_image.astype(np.uint8))
            for ip in IP:
                mask = (ip.sum(axis=-1) == 0)[..., np.newaxis].astype(np.uint8)
                new_image = ip + image * mask
                Xip.append(new_image.astype(np.uint8))

        fooled_yr = 0
        avg_conf_ip = 0
        total = 0
        per_image_results = {}
        assert len(Xr) == len(Xip)
        for i, (xr, xip) in enumerate(zip(Xr, Xip)):
            yr, conf_r = self.model.forward(xr, save_image=False, prediction_only=True)
            yip, conf_ip = self.model.forward(xip, save_image=False, prediction_only=True)
            per_image_results[i] = {
                'inert': xip, 'adversarial': xr, 'fooled': yr == adv_example_class, 'inert_conf': conf_ip,
                'adv_conf': conf_r}
            if yr == adv_example_class:
                fooled_yr += 1
            total += 1
            avg_conf_ip += conf_ip

        avg_conf_ip /= total
        return fooled_yr, avg_conf_ip, total, per_image_results
    
    @classmethod
    def generate_curve(cls, confidences, fooled_percentages, number_of_bins=25):
        """
        Algorithm 4 in SentiNet paper (https://arxiv.org/pdf/1812.00292v4.pdf).

        Given two arrays containing the per-sample confidences and fooled_percentages
        (returned by SentiNet Algorithm 3), digitizes confidences into a discrete
        number_of_bins, then uses the per-bin maximum confidence and fooled_percentage
        as a set of points for fitting a quadratic function. For each point that lies
        above the curve also computes an average distance of 
        these points from the curve, namely threshold.

        The curve has to be fit on benign samples.
        At test time, a point (x,y) is considered adversarial if
            y > curve(x) & dist((x,y),(x,curve(x)) > threshold ?

        Args:
            confidences (list[float]): the confidences for a set of samples
            fooled_percentages (list[float]): the fooled percentages for a set of samples
            number_of_bins (int): the number of bins to use for binning
        Returns:
            (func): a function with the fitted curve
            (threshold): the threshold above the curve

        """
        assert len(confidences) == len(fooled_percentages)

        confidences = np.asarray(confidences)
        fooled_percentages = np.asarray(fooled_percentages)

        # number of bins used to digitize the confidences
        number_of_bins = 25
        min_conf = confidences.min()
        max_conf = confidences.max()

        # make number_of_bins bins from min to max confidence
        bins = np.arange(min_conf, max_conf, (max_conf - min_conf)/number_of_bins)

        # save which bin each sample belongs to in a new column
        bin_identifier = np.digitize(confidences, bins)

        xs = []
        ys = []
        for i, _ in enumerate(bins):
            # filter only the samples that belong to this confidence bin
            # (offset by one because of np.digitise above)
            bin_mask = bin_identifier == i+1
            bin_conf, bin_fp = confidences[bin_mask], fooled_percentages[bin_mask]
            if bin_conf.shape[0] > 0:
                # get largest FoolPercentage
                i_of_max_fp = np.argmax(bin_fp)
                xs.append(bin_conf[i_of_max_fp])
                ys.append(bin_fp[i_of_max_fp])

        # fit curve to per-bin points
        curve_params, covariance = optimize.curve_fit(quadratic_func, np.array(xs), np.array(ys))

        avg_d = 0
        curve_func = lambda x: quadratic_func(x, *curve_params)

        # compute threshold 
        for x, y in zip(xs, ys):
            # NB. there is a mismatch between the code for Algorithm 4 in 
            # the SentiNet paper and the text. In the following if statement
            # we use the interpretation found in the text.
            if curve_func(x) < y:  # for each point lying above the curve
                # create function that returns 2D distance from (x,y) and (a, f(a))
                distance_func = lambda a: np.sqrt((x-a)**2+(y-curve_func(a))**2)
                # find minimum value for a
                result = optimize.minimize(distance_func, np.array(x), method='COBYLA')
                distance = distance_func(result['x'])
                # accumulate distances of points above curve
                avg_d += distance

        d = avg_d / confidences.shape[0]
        return curve_func, d
    
    @classmethod
    def is_adversarial(cls, curve, threshold, point):
        """
        Given curve describing a (quadratic) function 
        a threshold and a point = (x, y) returns True if 
            y > curve(x) & dist((x,y),(x,curve(x)) > threshold
        returns False otherwise.
        """
        x, y = point
        y_prime = curve(x)
        is_adv = False
        if y > y_prime:
            # we are above the curve, now calculate distance to curve
            distance_func = lambda a: np.sqrt((x-a)**2+(y-curve(a))**2)
            result = optimize.minimize(distance_func, x, method='COBYLA')
            distance = distance_func(result['x'])
            is_adv = distance > threshold
        return is_adv

    def run_sentinet(self, image_file, threshold, test_imgpaths, num_candidates=5, saliency='xrai', pattern='noise'):

        if saliency == 'xrai':
            saliency_func = self.model.xrai
        else:
            raise ValueError('Unsupported saliency function, choose one of %s' % ["xrai"])

        img = self.model.load_image(image_file)

        prediction, class_proposals = self.class_proposal(img, num_candidates)
        masks, mask_y = self.mask_generation(img, prediction, class_proposals, threshold=threshold,
                                             saliency_function=saliency_func)

        fooled_yr, avg_conf_ip, total, _ = self.testing(img, prediction[0], masks, test_imgpaths, pattern=pattern)
        return fooled_yr / total, avg_conf_ip


def run_wrap(sentinet, image_paths, reference_img_paths, threshold, candidates, saliency, pattern):
    results = []
    for image_path in tqdm(image_paths):
        fooled_percentage, confidence = sentinet.run_sentinet(image_path, threshold, reference_img_paths, candidates,
                                                              saliency=saliency, pattern=pattern)
        results.append((fooled_percentage, confidence, image_path, pattern, saliency))
    df = pd.DataFrame(results, columns=['FoolPercentage', 'Confidence', 'FileName', "Pattern", "Saliency"])
    return df


DS_MAP = {
    "gtsrbcnn": "gtsrb",
    "lisacnn": "lisa",
}

DET_MAP = {
    "gtsrbcnn": LisaCNNModel,
    "lisacnn": GtsrbCNNModel
}


def images_in_folder(folder, formats=("jpg", "jpeg", "png",)):
    files = os.listdir(folder)
    allowed_files = list(filter(lambda x: x.split(".")[-1] in formats, files))
    allowed_filepaths = [os.path.join(folder, x) for x in allowed_files]
    return allowed_filepaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', choices=["lisacnn_cvpr18", "gtsrbcnn_cvpr18"])
    parser.add_argument('-t', "--reference_imgs_folder", action="store")
    parser.add_argument('-b', "--test_benign_imgs_folder", action="store")
    parser.add_argument('-a', "--test_adversarial_imgs_folder", action="store")
    parser.add_argument('-o', "--output_folder", action="store", required=True)
    parser.add_argument('--threshold', help='Threshold for saliency', default=0.25, type=float)
    parser.add_argument('--saliency', help='Saliency algorithm to use', default='xrai', choices=["xrai"])
    parser.add_argument('--candidates', help='Specify the number of candidate classes', default=5, type=int)
    parser.add_argument("-p", '--pattern', choices=["checker", "noise"], default="noise", action="store")
    args = parser.parse_args()

    model_arch, model_id = args.model.split("_")[0], "_".join(args.model.split("_")[1:])
    ds_name = DS_MAP[model_arch]
    model = DET_MAP[model_arch](model_id1=model_id, saliency=True)

    p = yaml.load(open("/home/code/defences/params.yaml", "r"), yaml.FullLoader)

    np.random.seed(42)

    reference_imgs_fps = images_in_folder(args.reference_imgs_folder)
    test_benign_imgs_fps = images_in_folder(args.test_benign_imgs_folder)
    test_adversarial_imgs_fps = images_in_folder(args.test_adversarial_imgs_folder)

    sentinet = Sentinet(model)

    results = []

    # benign images
    df = run_wrap(sentinet, test_benign_imgs_fps, reference_imgs_fps,
                  args.threshold, args.candidates, args.saliency, args.pattern)
    df.to_csv(os.path.join(args.output_folder, 'benign_results.csv'))

    # benign images
    df = run_wrap(sentinet, test_adversarial_imgs_fps, reference_imgs_fps,
                  args.threshold, args.candidates, args.saliency, args.pattern)
    df.to_csv(os.path.join(args.output_folder, 'adversarial_results.csv'))
