import os

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils


THIS_PATH = os.path.dirname(os.path.realpath(__file__))

LABEL_KEYS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

FEATURE_KEYS = ["image_raw", "label"]

_LABEL_KEY = "label"

model_dir = os.path.join(THIS_PATH, 'models')


def _transformed_name(key):
    output = key
    if key == "image_raw":
        output = "input_1"  # input data name of keras inception-v3 model
    return output


def preprocessing_fn(inputs):
    outputs = {}
    for key in FEATURE_KEYS:
        outputs[_transformed_name(key)] = inputs[key]
    print('outputs in preprocessing', outputs)
    return outputs


def _get_raw_feature_spec(schema):
    # Tf.Transform considers these features as "raw"
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(filenames, tf_transform_output, batch_size=2):
    transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

    print('transformed_feature_spec', transformed_feature_spec)

    dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

    transformed_features = dataset.make_one_shot_iterator().get_next()
    # We pop the label because we do not want to use it as a feature while we're
    # training.
    return transformed_features, transformed_features.pop(_LABEL_KEY)


def serving_receiver_fn(tf_transform_output, schema):
    """Build the serving in inputs."""

    raw_feature_spec = _get_raw_feature_spec(schema)
    print("raw_feature_spec", raw_feature_spec)
    raw_feature_spec.pop(_LABEL_KEY)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    transformed_features = tf_transform_output.transform_raw_features(
        serving_input_receiver.features)

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)


def trainer_fn(hparams, schema):
    train_batch_size = 5
    eval_batch_size = 5

    '''
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_1": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)
    '''
    print('Hyperparameters in trainer_fn', hparams.__dict__)

    tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

    train_input_fn = lambda: input_fn(
        hparams.train_files,
        tf_transform_output,
        batch_size=train_batch_size)

    eval_input_fn = lambda: input_fn(
        hparams.eval_files,
        tf_transform_output,
        batch_size=eval_batch_size)

    export_serving_receiver_fn = serving_receiver_fn(tf_transform_output, schema)
    exporter = tf.estimator.FinalExporter('ml-pipeline', export_serving_receiver_fn)

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=hparams.train_steps)

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter])

    estimator = build_estimator()

    return {
        "estimator": estimator,
        "train_spec": train_spec,
        "eval_spec": eval_spec
    }


def build_estimator():
    inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
    print("Model input names", inception_v3.input_names)
    print("Model output names", inception_v3.output_names)
    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-5)
    inception_v3.compile(
        optimizer=optimiser,
        loss='categorical_crossentropy',
        metric='accuracy')

    estimator = tf.keras.estimator.model_to_estimator(keras_model=inception_v3)

    return estimator
