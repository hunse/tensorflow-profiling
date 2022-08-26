import datetime
import pathlib

import tensorflow as tf

file_dir = pathlib.Path(__file__).parent


def create_model(n_layers=5, layer_size=1000):
    x = inp = tf.keras.layers.Input(shape=(784,), dtype=tf.float32)
    for _ in range(n_layers):
        x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model([inp], [x])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def train_model(model, log_dir_name=None, batch_size=32, profile_batch=2):
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train.shape = x_test.shape = (-1, 28 * 28)

    n_train = 5 * batch_size
    x_train, y_train = x_train[:n_train], y_train[:n_train]

    # set up tensorboard callback
    if log_dir_name is None:
        log_dir_name = f"log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = file_dir / log_dir_name
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            profile_batch=profile_batch,
        )
    ]

    # Train the model
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=1,
        callbacks=callbacks,
    )

    print(f"Logged Tensorboard to: {log_dir}")
    return log_dir


def _run(log_dir_name, create_args, train_args):
    if not (file_dir / log_dir_name).exists():
        model = create_model(**create_args)
        train_model(model, log_dir_name=log_dir_name, **train_args)


def run_basic():
    _run(
        "log_basic",
        create_args=dict(n_layers=3, layer_size=2000),
        train_args=dict(batch_size=32),
    )


def run_profile5():
    _run(
        "log_profile5",
        create_args=dict(n_layers=3, layer_size=2000),
        train_args=dict(batch_size=32, profile_batch=(1, 5)),
    )


def run_high_op_count():
    # model with lots of ops and small workload per op, will have high
    _run(
        "log_high_op_count",
        create_args=dict(n_layers=20, layer_size=10),
        train_args=dict(batch_size=1),
    )


def run_all():
    run_basic()
    run_profile5()
    run_high_op_count()


run_all()
