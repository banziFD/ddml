import configparser
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.layers import dense

_INPUT_SIZE = [224, 224, 3]
DEFAULT_TYPE = tf.float32

def ddml_graph(config, input_holder1, input_holder2, pretrain_holder):
    assert input_holder1.shape.as_list()[1:] == _INPUT_SIZE
    assert input_holder1.shape.as_list()[1:] == _INPUT_SIZE

    with tf.name_scope("mobile_net") as scope:
        base_net = MobileNet(_INPUT_SIZE, alpha=1.0, include_top=False, pooling="avg", weights="imagenet")
        for i, layer in enumerate(base_net.layers):
            layer.trainable = True

    feature1 = base_net(input_holder1)
    feature2 = base_net(input_holder2)

    with tf.name_scope("classifier") as scope:
        nb_class = int(config["nb_class"])
        feature = base_net(pretrain_holder)
        logits = dense(feature, nb_class, activation=tf.math.sigmoid)

    return feature1, feature2, logits

def ddml_loss(config, feature1, feature2, label1, label2):
    # shape check
    assert feature1.shape.as_list() == feature2.shape.as_list()
    assert label1.shape.as_list() == label2.shape.as_list()
    assert feature1.shape.as_list == label1.shape.as_list()

    with tf.name_scope("ddml_loss") as scope:
        # load hyper parameter
        beta = float(config["beta"])
        tau = float(config["tau"])

        # loss
        sim = tf.dtypes.cast(tf.math.equal(label1, label2), dtype=tf.float32)
        sim =  sim * 2 - 1
        feature = feature1 - feature2
        distance = tf.norm(feature, axis = 1)
        loss = 1 - sim * (tau - distance)
        loss = (1 / beta) * tf.math.log(1 + tf.math.exp(beta * loss))

    return loss


if __name__ == "__main__":
    pass
