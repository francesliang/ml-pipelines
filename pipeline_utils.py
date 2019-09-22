import os

import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, InputLayer, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.datasets import mnist


THIS_PATH = os.path.dirname(os.path.realpath(__file__))

LABEL_KEYS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

FEATURE_KEYS = ["image_raw", "label"]

_LABEL_KEY = "label"

IM_SHAPE = (28, 28, 1)
DIM = (1, 28, 28, 1)

INPUT_LAYER = 'input_1'

model_dir = os.path.join(THIS_PATH, 'models')


def _transformed_name(key):
    output = key
    if key == "image_raw":
        output = INPUT_LAYER
    return output


def preprocessing_fn(inputs):
    outputs = {}
    for key in FEATURE_KEYS:
        print("input info", inputs[key])
        input_sparse_tensor = inputs[key]
        outputs[_transformed_name(key)] = input_sparse_tensor
    print('outputs in preprocessing', outputs)
    return outputs


def _get_raw_feature_spec(schema):
    # Tf.Transform considers these features as "raw"
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""

    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _example_proto_to_features_fn(example_proto):
    print('example_proto', example_proto)
    features = tf.parse_single_example(example_proto, features={
        'input_1': tf.FixedLenFeature([IM_SHAPE[0]*IM_SHAPE[1]*IM_SHAPE[2]], tf.int64),
        'label': tf.FixedLenFeature([10], tf.int64)
    })

    image = tf.cast(features['input_1'], tf.int32)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def _get_batch_iterator(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    dataset = dataset.map(_example_proto_to_features_fn)
    dataset = dataset.batch(batch_size).repeat().prefetch(1)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return {'input_1': features}, labels


'''
def input_fn(filenames, batch_size=2):
    file_dir = os.path.dirname(filenames[0])
    file_names=[os.path.join(file_dir, f) for f in os.listdir(file_dir)]
    features, labels = _get_batch_iterator(file_names, batch_size)
    print("input_fn features", features)
    print("input_fn labels", labels)
    return features, labels
'''

def input_fn(filenames, tf_transform_output, batch_size=2):
    transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

    print('transformed_feature_spec', transformed_feature_spec)

    #dataset = _gzip_reader_fn(filenames)

    dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

    transformed_features = dataset.make_one_shot_iterator().get_next()
    #features, labels = dataset.make_one_shot_iterator().get_next()
    #print('features in input-fn', features)

    #return {'input_1': features}, labels

    #transformed_features[INPUT_LAYER] = tf.sparse.reshape(transformed_features[key], DIM)

    print('transformed_features', transformed_features)
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

    print('serving_input_receiver.features', serving_input_receiver.features)

    transformed_features = tf_transform_output.transform_raw_features(
        serving_input_receiver.features)
    transformed_features.pop(_LABEL_KEY)

    print('transformed_features in serving', transformed_features)

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)


def eval_input_receiver_fn(tf_transform_output, schema):
    """Build everything needed for the tf-model-analysis to run the model."""
    raw_feature_spec = _get_raw_feature_spec(schema)
    serialized_tf_example = tf.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')

    # Add a parse_example operator to the tensorflow graph, which will parse
    # raw, untransformed, tf examples.
    features = tf.parse_example(serialized_tf_example, raw_feature_spec)

    # Now that we have our raw examples, process them through the tf-transform
    # function computed during the preprocessing step.
    transformed_features = tf_transform_output.transform_raw_features(
        features)

    # The key name MUST be 'examples'.
    receiver_tensors = {'examples': serialized_tf_example}

    #features.update(transformed_features)
    features = {INPUT_LAYER: transformed_features[INPUT_LAYER]}

    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=transformed_features[_LABEL_KEY])


def trainer_fn(hparams, schema):
    train_batch_size = 2
    eval_batch_size = 2

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

    export_serving_receiver_fn = lambda: serving_receiver_fn(tf_transform_output, schema)
    exporter = tf.estimator.FinalExporter('ml-pipeline', export_serving_receiver_fn)

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=hparams.train_steps)

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter])

    estimator = build_estimator()

    receiver_fn = lambda: eval_input_receiver_fn(
        tf_transform_output, schema)

    return {
        "estimator": estimator,
        "train_spec": train_spec,
        "eval_spec": eval_spec,
        "eval_input_receiver_fn": receiver_fn
    }


def build_estimator():
    '''
    num_classes = len(LABEL_KEYS)

    img_rows = 28
    img_cols = 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('x-train shape origianlly', x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    '''


    num_classes = len(LABEL_KEYS)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(IM_SHAPE[0]*IM_SHAPE[1]*IM_SHAPE[2],), name='input_1'))
    #x = tf.keras.Input(shape=(IM_SHAPE[0]*IM_SHAPE[1]*IM_SHAPE[2],), sparse=True, name='input_1')
    #x_ = tf.sparse_to_dense(x.indices, x.shape, x.values)
    model.add(tf.keras.layers.Reshape(IM_SHAPE))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=IM_SHAPE))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))


    tf_keras_model = model
    #optimiser = tf.keras.optimizers.Adam(learning_rate=1e-5)
    optimiser = tf.keras.optimizers.Adadelta()

    tf_keras_model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=optimiser)
                  #metrics=['accuracy'])

    estimator = tf.keras.estimator.model_to_estimator(keras_model=tf_keras_model)


    '''
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"conv2d_input": x_train},
        y=y_train,
        batch_size=100,
        shuffle=True,
        num_epochs=None)

    estimator.train(train_input_fn, steps=10)
    '''

    return estimator

