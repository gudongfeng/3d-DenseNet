import os
import random
import tempfile
import numpy as np
import PIL.Image as Image

from .base_provider import VideosDataset, DataProvider


class Data(VideosDataset):
  def __init__(self, paths, shuffle, normalization, sequence_length, 
               crop_size, num_classes):
    """
    Args:
      paths: list, list of string that have the video path and label 
        information
      shuffle: boolean, whether shuffle the data at the new epoch or not
      sequence_length: video clip length
      crop_size: image resize size
      normalization: `str` or None
        None: no any normalization
        divide_255: divide all pixels by 255
        divide_256: divide all pixels by 256
    """
    self.paths = paths
    self.shuffle = shuffle
    self.normalization = normalization
    self.sequence_length = sequence_length
    self.crop_size = crop_size
    self.num_classes = num_classes
    self.start_new_epoch()

  def start_new_epoch(self):
    self._batch_counter = 0
    if self.shuffle:
      random.shuffle(self.paths)

  @property
  def num_examples(self):
    return len(self.paths)

  def get_frames_data(self, filename, num_frames_per_clip=16):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays 
    
    Args
      num_frames_per_clip: sequence_length of the video clip 
    
    Returns
      video: numpy, video clip with shape 
        [sequence_length, crop_size, crop_size, channels]
    '''
    video = []
    s_index = 0
    for parent, dirnames, filenames in os.walk(filename):
      if(len(filenames) < num_frames_per_clip):
        return None
      filenames = sorted(filenames)
      s_index = random.randint(0, len(filenames) - num_frames_per_clip)
      for i in range(s_index, s_index + num_frames_per_clip):
        image_name = str(filename) + '/' + str(filenames[i])
        img = Image.open(image_name)
        img = img.resize((self.crop_size, self.crop_size))
        img_data = np.array(img).astype(float)
        if self.normalization:
          img_data = self.normalize_videos(img_data, self.normalization)
        video.append(img_data)
    return video

  def next_batch(self, batch_size):
    ''' Get the next batches of the dataset 
    Args
      batch_size: video batch size
    
    Returns
      videos: numpy, shape 
        [batch_size, sequence_length, crop_size, crop_size, channels]
      labels: numpy
        [batch_size, num_classes]
    '''
    start = self._batch_counter * batch_size
    self._batch_counter += 1
    videos_labels_slice = self.paths[start:]
    videos = []
    labels = []
    for line in videos_labels_slice:
      video_path, label = line.strip('\n').split()
      video = self.get_frames_data(video_path, self.sequence_length)
      if video is not None:
        videos.append(video)
        labels.append(int(label))
      if len(videos) is batch_size:
        videos = np.array(videos)
        # convert labels to one hot version
        labels = np.array(labels)
        labels = self.labels_to_one_hot(labels, self.num_classes)
        # return the videos and labels data
        return videos, labels

    # reach the end of the paths, start a new epoch
    self.start_new_epoch()
    return self.next_batch(batch_size)


class DataProvider(DataProvider):
  def __init__(self, num_classes, validation_set=None, shuffle=False,
               validation_split=None, normalization=None, crop_size=64,
               sequence_length=16, **kwargs):
    """
    Args:
      num_classes: the number of the classes
      validation_set: `bool`.
      validation_split: `int` or None
          float: chunk of `train set` will be marked as `validation set`.
          None: if 'validation set' == True, `validation set` will be
              copy of `test set`
      shuffle: `bool`, should shuffle data or not
      normalization: `str` or None
          None: no any normalization
          divide_255: divide all pixels by 255
          divide_256: divide all pixels by 256
      sequence_length: `integer`, video clip length
      crop_size: `integer`, the size that you want to reshape the images
    """
    self._num_classes = num_classes
    self._sequence_length = sequence_length
    self._crop_size = crop_size
    train_videos_labels = self.get_videos_labels_lines(
      'data_providers/train.list', shuffle)
    test_videos_labels = self.get_videos_labels_lines(
      'data_providers/test.list', shuffle)
    if validation_set and validation_split:
      random.shuffle(train_videos_labels)
      valid_videos_labels = train_videos_labels[:validation_split]
      train_videos_labels = train_videos_labels[validation_split:]
      self.validation = Data(valid_videos_labels, shuffle, 
                             normalization, sequence_length,
                             crop_size, num_classes)
    self.train = Data(train_videos_labels, shuffle, 
                      normalization, sequence_length,
                      crop_size, num_classes)
    self.test = Data(test_videos_labels, shuffle, 
                     normalization, sequence_length,
                     crop_size, num_classes)
    if validation_set and not validation_split:
      self.validation = self.test

  def get_videos_labels_lines(self, path, shuffle):
    # Open the file according to the filename
    lines = open(path, 'r')
    lines = list(lines)
    if shuffle:
      random.shuffle(lines)
    return lines

  @property
  def data_shape(self):
    return (self._sequence_length, self._crop_size, self._crop_size, 3)

  @property
  def n_classes(self):
    return self._num_classes