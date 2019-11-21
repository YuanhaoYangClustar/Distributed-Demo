import gzip
import shutil
import argparse
import tensorflow as tf


def create_model(data_format):
    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
        assert data_format == 'channels_last'
        input_shape = [28, 28, 1]

    l = tf.keras.layers
    max_pool = l.MaxPooling2D(
        (2, 2), (2, 2), padding='same',
        data_format=data_format)
    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(28 * 28,)),
            l.Conv2D(32, 5,
                     padding='same',
                     data_format=data_format,
                     activation=tf.nn.relu),
            max_pool,
            l.Conv2D(64, 5,
                     padding='same',
                     data_format=data_format,
                     activation=tf.nn.relu),
            max_pool,
            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.Dropout(0.4),
            l.Dense(10)
        ])


def model_fn(features, labels, mode, params):
    model = create_model('channels_first')
    learning_rate = params['learning_rate']
    image = features

    if isinstance(image, dict):
        image = features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }

        export_outputs = {
            'classify': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)

        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))

        tf.identity(learning_rate, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        tf.summary.scalar('train_accuracy', accuracy[1])
        train_op = optimizer.minimize(
            loss, tf.train.get_or_create_global_step())

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(logits, axis=1)
            )
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=eval_metric_ops)


def dataset(directory, images_file, labels_file):
    def decode_image(image):
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)
        label = tf.reshape(label, [])
        return tf.to_int32(label)

    dataset = tf.data.FixedLengthRecordDataset(
        directory + '/' + images_file,
        28 * 28, header_bytes=16)

    images = dataset.map(decode_image)
    dataset = tf.data.FixedLengthRecordDataset(
        directory + '/' + labels_file, 1, header_bytes=8)
    labels = dataset.map(decode_label)

    return tf.data.Dataset.zip((images, labels))


def train_dataset(directory, batch_size):
    train_img_zip = directory + '/train-images-idx3-ubyte.gz'
    train_label_zip = directory + '/train-labels-idx1-ubyte.gz'

    train_img_file = directory + '/train-images-idx3-ubyte'
    train_label_file = directory + '/train-labels-idx1-ubyte'

    if not tf.gfile.Exists(train_label_file):
        tf.logging.info('begin to unzip train examples of dataset')
        f_in = gzip.open(train_img_zip, 'rb')
        f_out = tf.gfile.Open(train_img_file, 'wb')
        shutil.copyfileobj(f_in, f_out)
        f_in.close()
        f_out.close()

        f_in = gzip.open(train_label_zip, 'rb')
        f_out = tf.gfile.Open(train_label_file, 'wb')
        shutil.copyfileobj(f_in, f_out)
        f_in.close()
        f_out.close()
        tf.logging.info('finish unzip')

    ds = dataset(directory,
                 'train-images-idx3-ubyte',
                 'train-labels-idx1-ubyte')
    ds = ds.cache().shuffle(buffer_size=50000).batch(batch_size)
    return ds


def eval_dataset(directory, batch_size):
    eval_img_zip = directory + '/t10k-images-idx3-ubyte.gz'
    eval_label_zip = directory + '/t10k-labels-idx1-ubyte.gz'

    eval_img_file = directory + '/t10k-images-idx3-ubyte'
    eval_label_file = directory + '/t10k-labels-idx1-ubyte'

    if not tf.gfile.Exists(eval_label_file):
        tf.logging.info('begin to unzip validate examples of dataset')
        f_in = gzip.open(eval_img_zip, 'rb')
        f_out = tf.gfile.Open(eval_img_file, 'wb')
        shutil.copyfileobj(f_in, f_out)
        f_in.close()
        f_out.close()

        f_in = gzip.open(eval_label_zip, 'rb')
        f_out = tf.gfile.Open(eval_label_file, 'wb')
        shutil.copyfileobj(f_in, f_out)
        f_in.close()
        f_out.close()
        tf.logging.info('finish unzip')

    ds = dataset(directory,
                 't10k-images-idx3-ubyte',
                 't10k-labels-idx1-ubyte')
    ds = ds.cache().shuffle(buffer_size=50000).batch(batch_size)
    # ds = ds.repeat(10000000)

    return ds


def setup_cmd_argument(parser):
    parser.add_argument(
        '--batch_size',
        dest='batch_size', type=int, default=256,
        help="batch size of all GPU in a worker")

    parser.add_argument(
        '--lr',
        dest='lr', type=float, default=1e-4,
        help="learning_rate during train")


if __name__ == '__main__':
    os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 1}
})
    tf.logging.set_verbosity(tf.logging.INFO)
    # setup command line argument and parse it.
    parser = argparse.ArgumentParser(
        description='mnist command line parser')
    setup_cmd_argument(parser)
    args = parser.parse_args()

    # prepare for train input and evalu input.
    data_dir = './dataset/'
    def eval_input_fn(): return eval_dataset(data_dir, args.batch_size)
    def train_input_fn(): return train_dataset(data_dir, args.batch_size)

    # create estimator
    params = {
        'learning_rate': args.lr
    }

    distribute = tf.contrib.distribute.CollectiveAllReduceStrategy(
            num_gpus_per_worker = 1)
    run_config = tf.estimator.RunConfig(
        model_dir='/tmp/mnist/',
        train_distribute=distribute)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params)

    # start train
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(
        estimator, train_spec, eval_spec)
