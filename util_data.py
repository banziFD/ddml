import os
import pickle
import glob
import json
import numpy as np
from PIL import Image
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _tfrecord_io(image_path, label_dict, keys, output_file):
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for key in keys:
            try:
                img = open(os.path.join(image_path, "{}.jpg".format(key)), 'rb').read()
                # width, height = Image.open(os.path.join(image_path, "{}.jpg".format(image))).size
            except FileNotFoundError:
                print("missing image {} !".format(key))
                continue
            lbl = int(label_dict[str(key)])
            image_id = str(key)

            img = _bytes_feature(img)
            lbl = _int64_feature(lbl)
            image_id = _bytes_feature(str.encode(image_id))
            # width = _int64_feature(width)
            # height = _int64_feature(height

            feature = {
                "image": img,
                "label": lbl,
                # "width": width,
                # "height": height
                "image_id": image_id}

            features = tf.train.Features(feature = feature)
            example = tf.train.Example(features = features)
            record_writer.write(example.SerializeToString())
        record_writer.close()

def parse_example_proto(example):
    feature_description = {"image": tf.FixedLenFeature([], tf.string),
                      "label": tf.FixedLenFeature([], tf.int64, default_value = -1),
                    #   "width": tf.FixedLenFeature([], tf.int64, default_value = -1),
                    #   "height": tf.FixedLenFeature([], tf.int64, default_value = -1),
                    "image_id": tf.FixedLenFeature([], tf.string)
                      }
    return tf.parse_single_example(example, feature_description)

def data2tfrecord(config):

    output_path = config["output_path"]
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    else:
        flag = input("{} exists press 'y' to clear:\n".format(output_path))
        if flag == 'y':
            shutil.rmtree(output_path)
            os.makedirs(output_path)

    image_path = config["image_path"]

    _tfrecord_io(image_path, label_dict, train_keys, os.path.join(output_path, "train.tfrecords"))
    _tfrecord_io(image_path, label_dict, val_keys, os.path.join(output_path, "val.tfrecords"))
    _tfrecord_io(image_path, label_dict, test_keys, os.path.join(output_path, "test.tfrecords"))

    summary = dict()
    summary["dataset_size"] = len(keys)
    summary["trainset_size"] = len(train_keys)
    summary["valset_size"] = len(val_keys)
    summary["testset_size"] = len(test_keys)
    summary["ignored_image_list"] = sorted(ignore_list)

    json.dump(summary, open(os.path.join(output_path, "{}_summary.json").format(time_stamp), 'w'))
    json.dump(train_keys, open(os.path.join(output_path, "{}_train_list.json".format(time_stamp)), 'w'))
    json.dump(test_keys, open(os.path.join(output_path, "{}_test_list.json".format(time_stamp)), 'w'))
    json.dump(val_keys, open(os.path.join(output_path, "{}_val_list.json".format(time_stamp)), 'w'))
    json.dump(label_dict, open(os.path.join(output_path, "{}_label_dict.json".format(time_stamp)), 'w'))
    print("complete!")

def _txt2list(metafile):
    with open(metafile, 'r') as fp:
        image_list = fp.read()
        image_list = image_list.split()
    return image_list

def _cifar_unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding="bytes")
    return d

def preprocess_cifar100_data(dataset_path):
    train_file = os.path.join(dataset_path, "train")
    test_file = os.path.join(dataset_path, "test")
    train_file = _cifar_unpickle(train_file)
    test_file = _cifar_unpickle(test_file)

    train_label = train_file[b"fine_labels"]
    train_image = train_file[b"data"]

    test_label = test_file[b"fine_labels"]
    test_image = test_file[b"data"]

    train_image = train_image.reshape(train_image.shape[0], 3, 32, 32)
    train_image = train_image.transpose(train_image, [0, 2, 3, 1])
    test_image = test_image.reshape(test_image.shape[0], 3, 32, 32)
    test_image = test_image.transpose(test_image, [0, 2, 3, 1])

    label_dict = dict()
    output_path = os.path.join(output_path)
    for i in range(train_image.shape[0]):
        label_dict = train_label[i]
        img = train_image[i]
        img = Image.fromarray(img)
        img.save(os.path.join(output_path, "{}.png".format(i)))

    train_size = train_image.shape[0]

    for i in range(test_image.shape[0]):
        label_dict = test_label[i]
        img = test_image[i]
        img = Image.fromarray(img)
        img.save(os.path.join(output_path, "{}.png".format(i + train_size)))

    json.dump(train_label_dict, open(os.path.join(dataset_path, "train_label.json"), 'w'))
    json.dump(test_label_dict, open(os.path.join(dataset_path, "test_label.json"), 'w'))

def preprocess_train(value):
    image, label = value["image"], value["label"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(image, (224, 224))
    return image, label

def preprocess_val(value):
    image, label = value["image"], value["label"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(image, (224, 224))
    return image, label

def preprocess_test(value):
    image, label, image_id = value["image"], value["label"], value["image_id"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(image, (224, 224))
    return image, label, image_id

def fetch_data(dataset, config, flag="train"):
    num_cpu = int(config["num_cpu"])
    batch_size = int(config["batch_size"])

    dataset = dataset.map(parse_example_proto, num_parallel_calls=num_cpu)
    if flag == "train":
        dataset = dataset.map(preprocess_train, num_parallel_calls=num_cpu)
        dataset = dataset.shuffle(1000)
        # dataset = dataset.repeat()
    elif flag == "val":
        dataset = dataset.map(preprocess_val, num_parallel_calls=num_cpu)
        # dataset = dataset.repeat()
    elif flag == "test":
        dataset = dataset.map(preprocess_test, num_parallel_calls=num_cpu)
    else:
        raise ValueError("flag should be in {train, val, test}")

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset

def get_data(config, flag):
    if flag in ("train", "val", "test"):
        summary = json.load(open(os.path.join(config["dataset_summary"]), 'r'))
        record_config_key = "{}_record".format(flag)
        data = tf.data.TFRecordDataset(config[record_config_key])
        data = fetch_data(data, config, flag = flag)
    else:
        raise ValueError("flag shoule be in {train, val, test}")
    return data, summary
