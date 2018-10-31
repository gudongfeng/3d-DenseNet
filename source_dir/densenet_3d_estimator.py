import os

import tensorflow as tf

from .densenet_3d_model import DenseNet3D


def model_fn(features, labels, mode, params):
    # Define the model
    model = DenseNet3D(
        video_clips=features['video_clips'], labels=labels, **params)

    # Get the prediction result
    if mode == tf.estimator.ModeKeys.PREDICT:
        model.is_training = False
        return _predict_result(model.logits)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=model.losses,
        train_op=model.train_op,
        eval_metric_ops={'eval_accuracy': model.accuracy})


def _predict_result(model):
    predictions = {'probabilities': model.prediction, 'logits': model.logits}
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)


def serving_input_fn(params):
    inputs = {
        'video_clips':
        tf.placeholder(
            tf.float32,
            shape=[
                None, params['num_frames_per_clip'], params['height'],
                params['width'], params['channel']
            ])
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(inputs)()


def train_input_fn(training_dir, params):
    directory = os.path.join(training_dir, 'train.tfrecord')
    return _build_tfrecord_dataset(directory, params['train_total_video_clip'],
                                   **params)


def eval_input_fn(evaluating_dir, params):
    directory = os.path.join(evaluating_dir, 'eval.tfrecord')
    return _build_tfrecord_dataset(directory, params['eval_total_video_clip'],
                                   **params)


def _build_tfrecord_dataset(directory, total_clip_num, batch_size, **params):
    '''
    Buffer the training dataset to TFRecordDataset with the following video shape
    [num_frames_per_clip, height, width, channel]
    ex: [16, 100, 120, 3]
    '''
    dataset = tf.data.TFRecordDataset(directory)
    dataset = dataset.shuffle(buffer_size=total_clip_num)
    dataset = dataset.map(
        map_func=
        lambda serialized_example: _parser(serialized_example, **params))
    dataset = dataset.repeat()
    iterator = dataset.batch(batch_size=batch_size).make_one_shot_iterator()
    clips, labels = iterator.get_next()
    return {'video_clips': clips}, labels


def _parser(serialized_example, num_frames_per_clip, **params):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'clip/width': tf.FixedLenFeature([], tf.int64),
            'clip/height': tf.FixedLenFeature([], tf.int64),
            'clip/channel': tf.FixedLenFeature([], tf.int64),
            'clip/raw': tf.FixedLenFeature([num_frames_per_clip], tf.string),
            'clip/label': tf.FixedLenFeature([], tf.int64)
        })

    def mapping_func(image):
        return _decode_image(image, **params)

    clip = tf.map_fn(mapping_func, features['clip/raw'], dtype=tf.float32)
    return clip, features['clip/label']


def _decode_image(image, channel, width, height, **params):
    image = tf.image.decode_jpeg(image, channels=channel)
    # This set_shape step is necesary for the last trainsition_layer_to_classes layer in the model
    image.set_shape([height, width, channel])
    image = tf.cast(image, tf.float32)
    return image
