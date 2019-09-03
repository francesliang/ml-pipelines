import os
from tfx.utils.dsl_utils import tfrecord_input
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen


def create_pipelines(tfrecord_path):

    examples = tfrecord_input(tfrecord_path)
    example_gen = ImportExampleGen(input_base=examples)

