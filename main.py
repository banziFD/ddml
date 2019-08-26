import os
import time
import datetime
import configparser
import argparse
import numpy as np
import tensorflow as tf

from util_model import ddml_graph
from util_model import ddml_loss
from util_data import get_data
from util_data import data2tfrecord
from util_data import cifar_preprocess

DEFAULT_TYPE = tf.float32
DATASET = ("cifar10, cifar100, imagenet, mscoco")

def ddml_train(config):

    # create tfrecord

    # load in dataset
    train_set1 = get_data(config, "train")
    val_set1 = get_data(config, "val")
    train_set2 = get_data(config, "train")
    val_set2 = get_data(config, "val")

    # make iterator to go over set
    iterator1 = tf.data.Iterator.from_structure(train_set1.output_types, train_set.output_shapes)
    iterator2 = tf.data.Iterator.from_structure(train_set2.output_types, train_set.output_shapes)

    train_init_op1 = iterator1.make_initializer(train_set1)
    val_init_op1 = iterator1.make_initializer(val_set1)
    train_init_op2 = iterator2.make_initializer(train_set2)
    val_init_op2 = iterator2.make_initializer(val_set2)
    next_element1 = iterator1.get_next()
    next_element2 = iterator2.get_next()
    image1, label1 = next_element1
    image2, label2 = next_element2

    # dummy holder to create computational graph
    pretrain_holder = tf.placeholder(DEFAULT_TYPE, [None, 224, 224, 3])

    feature1, feature2, logits = ddml_graph(image1, image2, pretrain_holder)
    loss = ddml_loss(config, feature1, feature2, label1, label2)
    optimizer = tf.train.AdamOptimizer(learning_rate=float(config["lr"]))
    updates = optimizer.minimize(loss)

    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(os.path.join(config["board"], "train"), graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(os.path.join(config["board"], "val"))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(config["epoch"]):
            print("training in {} epoch...".format(epoch))
            start = time.time()

            # training
            sess.run(train_init_op1)
            sess.run(train_init_op2)
            epoch_loss_train = 0
            step = 0
            try:
                while(True):
                    loss_current, _ = sess.run([loss, updates])
                    loss_current = np.sum(loss_current)
                    train_writer.add_summary(loss_current)
                    epoch_loss_train += loss_current
                    step += 1
            except tf.errors.OutOfRangeError:
                break
            epoch_loss_train /= step
            epoch_loss = epoch_loss_train
            train_writer.add_summary(epoch_loss, epoch)

            # validation
            sess.run(val_init_op1)
            sess.run(val_init_op2)
            epoch_loss_val = 0
            step = 0
            try:
                while(True):
                    loss_current = sess.run(loss)
                    loss_current = np.sum(loss_current)
                    epoch_loss_val += loss_current
                    step += 1
            except tf.errors.OutOfRangeError:
                break
            epoch_loss_val /= step
            epoch_loss = epoch_loss_val
            val_writer.add_summary(epoch_loss, epoch)

    return None


def ddml_pretrain():
    pass

def ddml_feature():
    pass


def main(config_file, process):
    config = configparser.ConfigParser()
    config.read(config_file)
    print(process)
    input()
    if process == "preprocess":
        dataset = input("which dataset are you currently working on?")
        if dataset in DATASET:
            config_pre = config["preprocess"]
            config_nn = config["nn"]
            if dataset == "cifar100":
                cifar_preprocess(config_pre, config_nn)

        else:
            raise ValueError("Unsupported dataset!")
    elif process == "train":
        pass
    elif process == "test":
        pass
    elif process == "feature":
        pass
    else:
        raise ValueError("Unsupported process!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for neural network")
    parser.add_argument("-config", help="configfile", default="./config.ini")
    parser.add_argument("-process", help="select process of neural network", default="train")
    args = parser.parse_args()

    main(args.config, args.process)