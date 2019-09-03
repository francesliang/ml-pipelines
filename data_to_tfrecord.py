import os
import sys
import json


def get_image_labels(image_dir, out_file='image_lables.json'):
    image_labels = {}
    for root, dirs, files in os.walk(image_dir):
        for label in dirs:
            for f in files:
                if not f.endswith('.png'):
                    continue
                fpath = os.path.join(root, label, f)
                image_labels[fpath] = label
    json.dump(image_labels, open(out_file, 'w'), indent=4)


if __name__ == '__main__':
    image_dir = sys.argv[1]
    get_image_labels(image_dir)


