import argparse
import json
import os
import sys
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans


def expand(df, output_filepath):
    min_number = 1
    nmps = df["n_matching_pixels"].values
    n_of_expanded_rows = nmps[nmps >= min_number].sum()
    mmap_triples = np.memmap(output_filepath, dtype=np.uint8, mode='w+', shape=(n_of_expanded_rows, 9))
    current_row_i = 0
    for i in range(df.shape[0]):
        row = df.iloc[i]
        nmp = int(row["n_matching_pixels"])
        if nmp >= min_number:
            xs_vals = row[["origin_r", "origin_g", 'origin_b', 'addition_r', 'addition_g', 'addition_b']].values
            ys_vals = row[["outcome_r", "outcome_g", 'outcome_b']].values
            row = np.hstack((xs_vals, ys_vals))
            stacked = np.tile(row, (nmp, 1))
            mmap_triples[current_row_i:current_row_i + nmp] = stacked
            current_row_i += nmp
        sys.stdout.write("\rPre-processing %d/%d" % (i, df.shape[0]))
        sys.stdout.flush()
    mmap_triples.flush()
    new_npy = mmap_triples.copy()
    np.save(output_filepath, new_npy)
    return True


def create_network(n_neurons_l1):

    with tf.compat.v1.variable_scope('proj_model'):
        X = tf.compat.v1.placeholder(tf.float32, [None, 6])
        Y = tf.compat.v1.placeholder(tf.float32, [None, 3])
        l1 = tf.keras.layers.Dense(n_neurons_l1, activation=tf.nn.relu, name="dense_layer1")(X)
        l2 = tf.keras.layers.Dense(3, activation=tf.nn.relu, name="dense_layer2")(l1)
        l3 = tf.keras.activations.tanh(l2)
    return l3, l2, l1, X, Y


def train(loss, xtrain, ytrain, xtest, ytest, opt_name, learning_rate, n_epochs, batch_size, conv_tol):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate) if opt_name == "gds" else tf.compat.v1.train.AdamOptimizer(learning_rate)
    optimizer_op = optimizer.minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch_i in range(n_epochs):

        for i in range(0, len(xtrain), batch_size):
            sess.run(optimizer_op, feed_dict={
                X: xtrain[i:i + batch_size],
                Y: ytrain[i:i + batch_size],
            })

        if epoch_i % 50 == 0:
            training_cost = sess.run(loss, feed_dict={X: xtrain, Y: ytrain})
            test_cost = sess.run(loss, feed_dict={X: xtest, Y: ytest})
            print('Epoch %d, train: %.4f, test: %.4f' % (epoch_i, training_cost, test_cost))
            if test_cost<=conv_tol and training_cost<=conv_tol:
                print("Converged, stopping")
                break

    return sess, training_cost, test_cost


def sample_output_space(sess, net_prediction, origin_color=None, origins=None, n=10000):

    org_r, org_g, org_b = np.random.rand(n, 3).T

    if origin_color is not None:
        org_r = np.ones(n) * origin_color[0]
        org_g = np.ones(n) * origin_color[1]
        org_b = np.ones(n) * origin_color[2]

    xs = np.vstack((org_r, org_g, org_b)).T

    if origins is not None:
        step = 10
        n = origins.shape[0]//step
        xs = origins[::step]
        if xs.shape[0]>n:
            xs = xs[:n]

    xs = np.concatenate([xs, np.random.rand(n, 3)], axis=1)
    return sess.run(net_prediction, feed_dict={X: xs})


def print_range(rs, gs, bs, name="empty"):
    print("%s -> Red output range: %.2f, %.2f" % (name, rs.min(), rs.max()))
    print("%s -> Green output range: %.2f, %.2f" % (name, gs.min(), gs.max()))
    print("%s -> Blue output range: %.2f, %.2f" % (name, bs.min(), bs.max()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="python projection_model.py -id <ID> -pm <LUX>")
    parser.add_argument("-id", "--experiment_id", type=str, required=True)
    parser.add_argument("-pm", "--proj_model_lux", type=str, required=True)
    parser.add_argument("-ob", "--object_id", type=str, required=True)
    parser.add_argument("-o", "--optimizer_name", type=str, choices=["gds", "adam"], default="adam")
    parser.add_argument("-e", "--n_epochs", type=int, default=1001)
    parser.add_argument("-b", "--batch_size", type=int, default=1000)
    parser.add_argument("-l", "--learning_rate", type=float, default=-1)
    parser.add_argument("-n1", "--number_of_neurons_in_l1", type=int, default=100)
    parser.add_argument("-tr", "--ratio_of_training_samples", type=float, default=0.9)
    parser.add_argument("-ct", "--convergence_tol", type=float, default=0.02)
    parser.add_argument("-r", "--redo_expansion", action="store_true", default=False)
    parser.add_argument("--test_network", default=True, action='store_true')
    args = parser.parse_args()

    # set default learning rate if one was not provided
    default_lr = {
        "adam": 0.001,
        "gds": 0.001
    }
    if args.learning_rate == -1:
        args.learning_rate = default_lr[args.optimizer_name]

    # paths and dirs
    this_script_name = os.path.basename(__file__).split(".")[0]
    experiment_folder = os.path.join("/home/data/%s" % args.experiment_id)
    output_folder = os.path.join(experiment_folder, this_script_name, args.proj_model_lux, args.object_id)
    os.makedirs(output_folder, exist_ok=True)
    profiling_folder = os.path.join(experiment_folder, "profile", args.proj_model_lux, args.object_id)
    profiling_params = {}
    if os.path.isfile(os.path.join(profiling_folder, "params.json")):
        profiling_params = json.load(open(os.path.join(profiling_folder, "params.json"), "r"))

    proj_model_params = vars(args)

    # Load in our data
    if args.redo_expansion or not os.path.isfile(os.path.join(output_folder, 'triples_expanded.npy')):
        df = pd.read_csv(os.path.join(profiling_folder, "all_triples.csv"))
        output_expanded = os.path.join(output_folder, 'triples_expanded.npy')
        _ = expand(df, output_expanded)

    mat = np.load(os.path.join(output_folder, 'triples_expanded.npy'))
    xs = mat[:, :6].astype(float)/255.0
    ys = mat[:, 6:].astype(float)/255.0
    xs, ys = shuffle(xs, ys)

    n_train = int(xs.shape[0]*args.ratio_of_training_samples)

    xs_train, ys_train = xs[:n_train], ys[:n_train]
    xs_test, ys_test = xs[n_train:], ys[n_train:]
    net_prediction, layer2, layer1, X, Y = create_network(args.number_of_neurons_in_l1)

    # make loss
    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(net_prediction - Y), axis=-1))
    loss_red = tf.reduce_mean(tf.abs(net_prediction - Y)[..., 0])
    loss_green = tf.reduce_mean(tf.abs(net_prediction - Y)[..., 1])
    loss_blue = tf.reduce_mean(tf.abs(net_prediction - Y)[..., 2])

    try:
        session, traincost_, testcost_ = train(loss, xs_train, ys_train, xs_test, ys_test,
                                               args.optimizer_name, args.learning_rate,
                                               args.n_epochs, args.batch_size, args.convergence_tol)
    except KeyboardInterrupt as e:
        print("Interrupted")
        exit()

    _, loss_red_, loss_green_, loss_blue_ = session.run(
        [net_prediction, loss_red, loss_green, loss_blue], feed_dict={X: xs, Y: ys})

    print("Per channel RGB loss: %.4f, %.4f, %.4f" % (loss_red_, loss_green_, loss_blue_))

    proj_model_params["train_cost"] = float(traincost_)
    proj_model_params["test_cost"] = float(testcost_)
    if float(testcost_)>0.03:
        print("################################################################")
        print("WARNING: MSE on test data is high (%.3f), aim for <0.03." % float(testcost_))
        print("################################################################")
    proj_model_params["train_cost_red"] = float(loss_red_)
    proj_model_params["train_cost_green"] = float(loss_green_)
    proj_model_params["train_cost_blue"] = float(loss_blue_)

    if args.test_network:
        # visualize how projection space looks like
        origins_to_test = [[.5, 0, 0], [.5, .5, .5]]
        origins_names = ["gray", "red"]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('red')
        ax.set_ylabel('green')
        ax.set_zlabel('blue')

        # plot for traindata
        predicted = session.run(net_prediction, feed_dict={X: xs})
        kmeans = KMeans(n_clusters=2, random_state=0).fit(predicted)

        output_plot_3d = []
        reds, greens, blues = [], [], []
        markers = ["o", "^"]
        for i in range(kmeans.n_clusters):
            output_samples = sample_output_space(session, net_prediction, origins=predicted[kmeans.labels_ == i])
            output_samples = np.clip(output_samples, 0, 1.0)
            r, g, b = output_samples.T
            reds.extend(r.tolist())
            greens.extend(g.tolist())
            blues.extend(b.tolist())
            print("Cluster %d" % i)
            print_range(r, g, b)
            ax.scatter(r, g, b, s=100, marker=markers[i], c=output_samples, label=origins_names[i], edgecolors="k", depthshade=0)

        plt.legend()
        output_plot_3d = pd.DataFrame(np.array([reds, greens, blues]).T, columns=["red", "green", "blue"])
        output_plot_3d.to_csv(os.path.join(output_folder, '3dplot_data.csv'), index=False)
        plt.show()
        fig.savefig(os.path.join(output_folder, 'proj_model_plot.pdf'))

    vars = tf.compat.v1.trainable_variables()
    vars_vals = session.run(vars)
    var_dict = {}
    for var, val in zip(vars, vars_vals):
        # print("var: {}, value: {}".format(var.name, val))
        var_dict[var.name] = val.tolist()

    json.dump(var_dict, open(os.path.join(output_folder, "projection_model.json"), "w"))
    json.dump(proj_model_params, open(os.path.join(output_folder, "params.json"), "w"), indent=2)
    print()


