import os
import sys
import json

from utils.data import write_tfrecords


def get_image_labels(image_dir, out_file='image_labels.json'):
    image_labels = {}
    labels = sorted(os.listdir(image_dir))
    print(labels)
    for root, dirs, files in os.walk(image_dir):
        for label in dirs:
            dir_path = os.path.join(root, label)
            for f in os.listdir(dir_path):
                if not f.endswith('.png'):
                    continue
                fpath = os.path.join(root, label, f)
                image_labels[fpath] = labels.index(label)
    json.dump(image_labels, open(out_file, 'w'), indent=4)
    return image_labels


if __name__ == '__main__':
    image_dir = sys.argv[1]
    tfrecord_file = sys.argv[2]
    image_labels = get_image_labels(image_dir)
    write_tfrecords(tfrecord_file, image_labels)

