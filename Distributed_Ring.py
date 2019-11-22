import tensorflow_datasets as tfds
import tensorflow as tf
import os, json

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {
#         "worker": ["10.11.4.2:20000", "10.11.0.2:20000"]
#     },
#     'task': {'type': 'worker', 'index': 0}
# })


def input_fn(mode, input_context=None):
    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True)
    mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                     datasets['test'])

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                            input_context.input_pipeline_id)
    return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


LEARNING_RATE = 1e-4


def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        from_logits=True, y_true=labels, y_pred=logits)
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    print("loss is ")
    print(loss)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()))


strategy = tf.contrib.distribute.CollectiveAllReduceStrategy()

config = tf.estimator.RunConfig(train_distribute=strategy)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)
