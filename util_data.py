import os

import glob
import numpy as np
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

def _divide(keys, train_val_test):
    length = len(keys)
    random.shuffle(keys)
    boundary1 = int(length * train_val_test[0])
    boundary2 = int(boundary1 + length * train_val_test[1])
    boundary3 = int(length * train_val_test[2])
    train_keys = keys[0: boundary1]
    val_keys = keys[boundary1: boundary2]
    test_keys = keys[-boundary3:]
    return train_keys, val_keys, test_keys

def data2tfrecord(config):
    time_stamp = datetime.datetime.utcnow().strftime("%s")

    label_dict = _get_combine_label(config)
    ignore_list = _get_combine_ignore(config)
    for image_id in ignore_list:
        label_dict.pop(image_id, None)

    keys = list(label_dict.keys())
    train_val_test = json.loads(config["train_val_test"])
    train_keys, val_keys, test_keys = _divide(keys, train_val_test)

    output_path = config["output_path"]
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    else:
        flag = input("{} exists press 'y' to clear:\n".format(output_path))
        if flag == 'y':
            shutil.rmtree(output_path)
            os.makedirs(output_path)

    image_path = config["image_path"]

    _tfrecord_io(image_path, label_dict, train_keys, os.path.join(output_path, "{}_train.tfrecords".format(time_stamp)))
    _tfrecord_io(image_path, label_dict, val_keys, os.path.join(output_path, "{}_val.tfrecords".format(time_stamp)))
    _tfrecord_io(image_path, label_dict, test_keys, os.path.join(output_path, "{}_test.tfrecords".format(time_stamp)))

    summary = dict()
    summary["dataset_size"] = len(keys)
    summary["trainset_size"] = len(train_keys)
    summary["valset_size"] = len(val_keys)
    summary["testset_size"] = len(test_keys)
    summary["ignored_image_list"] = sorted(ignore_list)
    summary["time_stamp"] = time_stamp

    json.dump(summary, open(os.path.join(output_path, "{}_summary.json").format(time_stamp), 'w'))
    json.dump(train_keys, open(os.path.join(output_path, "{}_train_list.json".format(time_stamp)), 'w'))
    json.dump(test_keys, open(os.path.join(output_path, "{}_test_list.json".format(time_stamp)), 'w'))
    json.dump(val_keys, open(os.path.join(output_path, "{}_val_list.json".format(time_stamp)), 'w'))
    json.dump(label_dict, open(os.path.join(output_path, "{}_label_dict.json".format(time_stamp)), 'w'))
    print("complete!")

def preprocess_train(value, classify=Flase):
    image, label = value["image"], value["label"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(image, (224, 224))
    if classify:
        label = tf.one_hot(label, 101)
    return image, label

def preprocess_val(value, classify=Flase):
    image, label = value["image"], value["label"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(image, (224, 224))
    if classify:
        label = tf.one_hot(label, 101)
    return image, label

def preprocess_test(value, classify=False):
    image, label, image_id = value["image"], value["label"], value["image_id"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(image, (224, 224))
    if classify:
        label = tf.one_hot(label, 101)
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
