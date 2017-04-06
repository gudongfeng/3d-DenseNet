import tempfile
import os
import PIL.Image as Image
#from six.moves import xrange

import numpy as np

from .base_provider import VideosDataset, DataProvider

class Data(VideosDataset):
  def __init__(self, videos, labels, shuffle, normalization):
    """
    Args:
      videos: 5D numpy array 
        [num_examples, sequence_length, height, weight, channels]
      labels: 2D or 1D numpy array
      shuffle: `bool`, should shuffle data or not
      normalization: `str` or None
        None: no any normalization
        divide_255: divide all pixels by 255
        divide_256: divide all pixels by 256
        by_channels: substract mean of every channel and divide each
          channel data by it's standart deviation
    """
    self.shuffle = shuffle
    self.videos = videos
    self.labels = labels
    self.normalization = normalization
    self.start_new_epoch()

  def start_new_epoch(self):
    self._batch_counter = 0
    if self.shuffle:
      self.videos, self.labels = self.shuffle_videos_and_labels(
        self.videos, self.labels
      )

  @property
  def num_examples(self):
    return self.labels.shape[0]

  def next_batch(self, batch_size):
    start = self._batch_counter * batch_size
    end = (self._batch_counter + 1) * batch_size
    self._batch_counter += 1
    videos_slice = self.videos[start:end]
    labels_slice = self.labels[start:end]
    # due to memory error it should be done inside batch
    if self.normalization is not None:
      videos_slice = self.normalize_videos(videos_slice, self.normalization)
    if videos_slice.shape[0] != batch_size:
      self.start_new_epoch()
      return self.next_batch(batch_size)
    else:
      return videos_slice, labels_slice


class DataProvider(DataProvider):
  def __init__(self, num_classes, validation_set=None, one_hot=True,
               validation_split=None, shuffle=False, normalization=None,
               sequence_length=16, overlap_length=8, crop_size=64, **kwargs):
    """
    Args:
      num_classes: the number of the classes
      validation_set: `bool`.
      validation_split: `int` or None
          float: chunk of `train set` will be marked as `validation set`.
          None: if 'validation set' == True, `validation set` will be
              copy of `test set`
      one_hot: `bool`, return lasels one hot encoded
      shuffle: `bool`, should shuffle data or not
      normalization: `str` or None
          None: no any normalization
          divide_255: divide all pixels by 255
          divide_256: divide all pixels by 256
          by_chanels: substract mean of every chanel and divide each
              chanel data by it's standart deviation
      sequence_length: `integer`, video clip length
      overlap_length: `integer`, the overlap of the images when we extract
        the video clips this should be less than sequence_length
      crop_size: `integer`, the size that you want to reshape the images
    """
    self._num_classes = num_classes
    self._sequence_length = sequence_length
    self._overlap_length = overlap_length
    self._crop_size = crop_size
    train_videos, train_labels = self.get_videos_and_labels('data_providers/train.list', one_hot)
    test_videos, test_labels = self.get_videos_and_labels('data_providers/test.list', one_hot)
    if validation_set and validation_split:
      rand_indexes = np.random.permutation(train_videos.shape[0])
      valid_indexes = rand_indexes[:validation_split]
      train_indexes = rand_indexes[validation_split:]
      valid_videos = train_videos[valid_indexes]
      valid_labels = train_labels[valid_indexes]
      train_videos = train_videos[train_indexes]
      train_labels = train_labels[train_indexes]
      self.validation = Data(
        valid_videos, valid_labels, shuffle, normalization)
    self.train = Data(
        train_videos, train_labels, shuffle, normalization)
    self.test = Data(
        test_videos, test_labels, shuffle, normalization)

    if validation_set and not validation_split:
      self.validation = self.test

  def _get_frames_data(self, filename):
    '''Given a directory containing extracted frames, return a video clip of
    sequence_length consecutive frames as a list of np arrays
    
    Args
      filename: video file path name
    Returns
      videos: list of video clips, length of video clip is sequence_length
    '''
    videos = []
    index = 0
    for parent, dirnames, filenames in os.walk(filename):
      while True:
        if index + self._sequence_length > len(filenames):
          # Add the last sequence_length of images as a video clip in the filename
          if len(filenames) >= self._sequence_length:
            start = len(filenames) - self._sequence_length
            video = []
            for i in range(start, len(filenames)):
              image_name = str(filename) + '/' + str(filenames[i])
              img = Image.open(image_name)
              img = img.resize((self._crop_size, self._crop_size))
              img_data = np.array(img)
              video.append(img_data)
            videos.append(video)
          return videos
        filenames = sorted(filenames)
        video = []
        for i in range(index, index + self._sequence_length):
          image_name = str(filename) + '/' + str(filenames[i])
          img = Image.open(image_name)
          img = img.resize((self._crop_size, self._crop_size))
          img_data = np.array(img)
          video.append(img_data)
        videos.append(video)
        # update the index
        index += (self._sequence_length - self._overlap_length)

  def get_videos_and_labels(self, name_part, one_hot=False):
    '''Get the video and label datas from the file

    Args
      name_part: `str`, should be the path of the data path file
      one_hot: `boolean`, whether convert the label to one hot version or not

    Returns
      data: `numpy`, [num_examples, sequence_length, height, weight, channel]
      labels: `numpy`, [num_examples] or [num_examples, num_classes]
    '''
    # Open the file according to the filename
    lines = open(name_part, 'r')
    lines = list(lines)
    data = []
    labels = []
    for index in range(len(lines)):
      # Get the video path information and the label information
      line = lines[index].strip('\n').split()
      dir_name = line[0]
      tmp_label = line[1]
      # tmp_data is a list of videos which contain numpy images
      tmp_data = self._get_frames_data(dir_name)
      if len(tmp_data) != 0:
        for video_index in range(len(tmp_data)):
          labels.append(int(tmp_label))
        data.extend(tmp_data)
    data = np.array(data)
    labels = np.array(labels)
    if one_hot:
      labels = self.labels_to_one_hot(labels)
    return data, labels

  @property
  def n_classes(self):
    return self._num_classes

  @property
  def data_shape(self):
    return (self._sequence_length, self._crop_size, self._crop_size, 3)


if __name__ == '__main__':
  # WARNING: this test will require about 5 GB of RAM
  import matplotlib.pyplot as plt

  def plot_images_labels(videos, labels, axes, main_label):
    plt.text(0, 1.5, main_label, ha='center', va='top',
         transform=axes[len(axes) // 2].transAxes)
    for video, label, axe in zip(videos, labels, axes):
      axe.imshow(video[0])
      axe.set_title(np.argmax(label))
      axe.set_axis_off()

  n_plots = 10
  fig, axes = plt.subplots(nrows=2, ncols=n_plots)

  dataset = DataProvider(5)
  videos, labels = dataset.train.next_batch(n_plots)
  plot_images_labels(
    videos,
    labels,
    axes[0],
    'Original dataset')

  dataset = DataProvider(5, shuffle=True)
  videos, labels = dataset.train.next_batch(n_plots)
  plot_images_labels(
    videos,
    labels,
    axes[1],
    'Shuffled dataset')

  plt.show()
