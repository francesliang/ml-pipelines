import sys
import os

from tfx import components
from tfx.utils.dsl_utils import tfrecord_input
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen


def create_pipelines(tfrecord_dir):

    examples = tfrecord_input(tfrecord_dir)
    example_gen = ImportExampleGen(input_base=examples)
    print('example-gen', example_gen)


if __name__ == '__main__':
    tfrecord_dir = sys.argv[1]
    create_pipelines(tfrecord_dir)

