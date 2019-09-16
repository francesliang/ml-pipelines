import os

import tensorflow as tf

THIS_PATH = os.path.dirname(os.path.realpath(__file__))

LABEL_KEYS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

model_dir = os.path.join(THIS_PATH, 'models')


def preprocessing_fn(inputs):
    outputs = {}
    outputs = inputs
    print('outputs in preprocessing', outputs)
    return outputs


def trainer_fn(hparams, schema):
    train_batch_size = 50
    eval_batch_size = 50

    '''
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_1": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)
    '''
    print('Hyperparameters in trainer_fn', hparams.__dict__)


    return {}


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
