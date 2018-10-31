import os
from io import BytesIO
from PIL import Image

import cv2
import tensorflow as tf

FLAGS = tf.app.flags
FLAGS.DEFINE_string('data_dir',
                    '/Users/dongfenggu/Desktop/action_kth/origin_videos',
                    'Path to the input data')
FLAGS.DEFINE_string('train_output_path',
                    'train.tfrecord',
                    'Path to output train TFRecord')
FLAGS.DEFINE_string('eval_output_path',
                    'eval.tfrecord',
                    'Path to output eval TFRecord')
FLAGS.DEFINE_float(
    'train_eval_split_factor', 0.75,
    'use this factor to split the train (default 3/4) and '
    'eval data (default 1/4) in data_dir')
FLAGS.DEFINE_integer('width', 120, 'customize image width')
FLAGS.DEFINE_integer('height', 100, 'customize image height')
FLAGS.DEFINE_integer('channel', 3, 'image color channel')
FLAGS.DEFINE_integer('skip_frames', 10,
                     'the number of frames we skip when we process the video')
FLAGS.DEFINE_integer('num_frames_per_clip', 16,
                     'the number of frames for a video clip')
FLAGS = FLAGS.FLAGS


def get_clips(image_list):
    # Given a list of images, return video clips of (num_frames_per_clip) consecutive frames as a list.
    video_clips = []
    images_len = len(image_list)
    if images_len < FLAGS.num_frames_per_clip:
        return video_clips

    # Prepare the first clip
    video_clips.append(image_list[:FLAGS.num_frames_per_clip])

    num_of_extra_clip = int(
        (images_len - FLAGS.num_frames_per_clip) / FLAGS.skip_frames)
    for i in range(1, num_of_extra_clip + 1):
        start = i * FLAGS.skip_frames - 1
        end = start + FLAGS.num_frames_per_clip
        video_clips.append(image_list[start:end])

    return video_clips


def process_dataset(train_writer, eval_writer, data_dir):
    label = -1
    # [class1, class2, class3, ..., class n]
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            # Set the label value for this class, start from 0
            label += 1
            print("Processing class: " + str(label))
            # Process each video file in this class
            video_filenames = os.listdir(class_path)
            for video_filename in video_filenames[0:int(
                    FLAGS.train_eval_split_factor * len(video_filenames))]:
                process_video(train_writer, class_path, video_filename, label)
            for video_filename in video_filenames[
                    int(FLAGS.train_eval_split_factor *
                        len(video_filenames)):len(video_filenames)]:
                process_video(eval_writer, class_path, video_filename, label)


def process_video(writer, class_path, video_filename, label):
    video_filename_path = os.path.join(class_path, video_filename)
    if video_filename_path.endswith('avi'):
        video_clips = _convert_video_to_clips(video_filename_path)
        # Convert the clip to tf record
        for clip in video_clips:
            tf_example = create_tf_example(raw=clip, label=label)
            writer.write(tf_example.SerializeToString())


def _convert_video_to_clips(video_path):
    # Use opencv to read video to list of images
    video_images_list = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        # frame shape [height, width, channel]
        _, frame = cap.read()
        try:
            # pil_image shape [width, height, channel]
            pil_image = Image.fromarray(frame)
            # Resize the image and convert the image according to the channel information
            if FLAGS.channel == 1:
                pil_image = pil_image.resize((FLAGS.width, FLAGS.height),
                                             Image.NEAREST).convert('L')
            else:
                pil_image = pil_image.resize((FLAGS.width, FLAGS.height),
                                             Image.NEAREST)
            # Encode the image to JPEG
            with BytesIO() as buffer:
                pil_image.save(buffer, format="JPEG")
                video_images_list.append(buffer.getvalue())
        except AttributeError:
            # Fail to read the image
            break

    # Convert list of images to clips of images with type np.float32
    return get_clips(image_list=video_images_list)


def _bytelist_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tf_example(raw, label):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'clip/width': _int64_feature(FLAGS.width),
                'clip/height': _int64_feature(FLAGS.height),
                'clip/channel': _int64_feature(FLAGS.channel),
                'clip/raw': _bytelist_feature(raw),
                'clip/label': _int64_feature(label)
            }))


def get_total_video_clip_number(data_path):
    count = 0
    for _ in tf.python_io.tf_record_iterator(data_path):
        count += 1
    return count


def main(_):
    # Write the dataset
    train_writer = tf.python_io.TFRecordWriter(FLAGS.train_output_path)
    eval_writer = tf.python_io.TFRecordWriter(FLAGS.eval_output_path)

    process_dataset(
        train_writer=train_writer,
        eval_writer=eval_writer,
        data_dir=FLAGS.data_dir)

    train_writer.close()
    eval_writer.close()

    # Count the dataset record
    print("Total clips in train dataset: " +
          str(get_total_video_clip_number(FLAGS.train_output_path)))
    print("Total clips in eval dataset: " +
          str(get_total_video_clip_number(FLAGS.eval_output_path)))


if __name__ == '__main__':
    tf.app.run()
