import os
import cv2
import random
import tempfile
import numpy as np
from Queue import Queue
from threading import Thread

from .base_provider import VideosDataset, DataProvider

class Data(VideosDataset):
  def __init__(self, name, paths, normalization, sequence_length,
               crop_size, num_classes, queue_size):
    """
    Args:
      name: str, name of the data (train, test or validation)
      paths: list, list of string that have the video path and label 
        information
      sequence_length: video clip length
      crop_size: `tuple`, image resize size (width, height)
      normalization: `str` or None
        None: no any normalization
        divide_255: divide all pixels by 255
        divide_256: divide all pixels by 256
      num_classes: `integer`, number of classes that the dataset has
      queue_size: `integer`, data queue size
    """
    self.name             = name
    self.paths            = paths
    self.normalization    = normalization
    self.sequence_length  = sequence_length
    self.crop_size        = crop_size
    self.num_classes      = num_classes
    self.queue            = DataQueue(name, queue_size)
    self.examples         = None
    self._start_data_thread()

  def get_frames_data(self, filename, num_frames_per_clip=16):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays
    
    Args
      num_frames_per_clip: sequence_length of the video clip
    
    Returns
      video: numpy, video clip with shape
        [sequence_length, width, height, channels]
    '''
    video = []
    s_index = 0
    for parent, dirnames, files in os.walk(filename):
      filenames = [ fi for fi in files if not fi.endswith((".png", ".jpg", "jpeg")) ]
      if(len(filenames) < num_frames_per_clip):
        return None
      suffix = filenames[0].split('.', 1)[1]
      filenames_int = [int(i.split('.', 1)[0]) for i in filenames]
      filenames_int = sorted(filenames_int)
      s_index = random.randint(0, len(filenames) - num_frames_per_clip)
      for i in range(s_index, s_index + num_frames_per_clip):
        image_name = str(filename) + '/' + str(filenames_int[i]) + '.' + suffix
        # print image_name
        img = cv2.imread(image_name)
        img = cv2.resize(img, self.crop_size)
        if self.normalization:
          img_data = self.normalize_image(img, self.normalization)
        video.append(img_data)
    return video

  def extract_video_data(self):
    ''' Single tread to extract video and label information from the dataset
    '''
    # Generate one randome index and 
    while True:
      index = random.randint(0, len(self.paths)-1)
      video_path, label = self.paths[index].strip('\n').split()
      video = self.get_frames_data(video_path, self.sequence_length)
      if video is not None and len(video) == self.sequence_length:
        # Put the video into the queue
        video = np.array(video)
        label = np.array(int(label))
        self.queue.put((video, label))

  def _start_data_thread(self):
    print("Start thread: %s data preparation ..." % self.name)
    self.worker = Thread(target=self.extract_video_data)
    self.worker.setDaemon(True)
    self.worker.start()

  @property
  def num_examples(self):
    if not self.examples:
      # calculate the number of examples
      total = 0
      for line in self.paths:
        video_path, _ = line.strip('\n').split()
        for root, dirs, files in os.walk(video_path):
          total += len(files)
      self.examples = total / self.sequence_length
    return self.examples

  def next_batch(self, batch_size):
    ''' Get the next batches of the dataset 
    Args
      batch_size: video batch size
    
    Returns
      videos: numpy, shape 
        [batch_size, sequence_length, width, height, channels]
      labels: numpy
        [batch_size, num_classes]
    '''
    videos, labels = self.queue.get(batch_size)
    videos = np.array(videos)
    labels = np.array(labels)
    labels = self.labels_to_one_hot(labels, self.num_classes)
    return videos, labels


class DataQueue():
  def __init__(self, name, maximum_item, block=True):
    """
    Args
      name: str, data type name (train, validation or test)
      maximum_item: integer, maximum item that this queue can store
      block: boolean, block the put or get information if the queue is
        full or empty
    """
    self.name         = name
    self.block        = block
    self.maximum_item = maximum_item
    self.queue        = Queue(maximum_item)

  @property
  def queue(self):
    return self.queue

  @property
  def name(self):
    return self.name

  def put(self, data):
    self.queue.put(data, self.block)

  def get(self, batch_size):
    '''
    Args:
      batch_size: integer, the number of the item you want to get from the queue
    
    Returns:
      videos: list, list of numpy data with shape
        [sequence_length, width, height, channels]
      labels: list, list of integer number
    '''
    videos = []
    labels = []
    for i in range(batch_size):
      video, label = self.queue.get(self.block)
      videos.append(video)
      labels.append(label)
    return videos, labels


class DataProvider(DataProvider):
  def __init__(self, num_classes, validation_set=None, test=False,
               validation_split=None, normalization=None, crop_size=(64,64),
               sequence_length=16, train_queue=None, valid_queue=None,
               test_queue=None, train=False, queue_size=300, **kwargs):
    """
    Args:
      num_classes: the number of the classes
      validation_set: `bool`.
      validation_split: `int` or None
          float: chunk of `train set` will be marked as `validation set`.
          None: if 'validation set' == True, `validation set` will be
              copy of `test set`
      normalization: `str` or None
          None: no any normalization
          divide_255: divide all pixels by 255
          divide_256: divide all pixels by 256
      sequence_length: `integer`, video clip length
      crop_size: `tuple`, the size that you want to reshape the images, (width, height)
      train: `boolean`, whether we need the training queue or not
      test: `test`, whether we need the testing queue or not
      queue_size: `integer`, data queue size , default is 300
    """
    self._num_classes = num_classes
    self._sequence_length = sequence_length
    self._crop_size = crop_size
    train_videos_labels = self.get_videos_labels_lines(
      'data_providers/train.list')
    test_videos_labels = self.get_videos_labels_lines(
      'data_providers/test.list')
    if validation_set and validation_split:
      random.shuffle(train_videos_labels)
      valid_videos_labels = train_videos_labels[:validation_split]
      train_videos_labels = train_videos_labels[validation_split:]
      self.validation = Data('validation', valid_videos_labels,
                             normalization, sequence_length,
                             crop_size, num_classes, queue_size)
    if train:
      self.train = Data('train', train_videos_labels,
                        normalization, sequence_length,
                        crop_size, num_classes, queue_size)
    if test:
      self.test = Data('test', test_videos_labels,
                      normalization, sequence_length,
                      crop_size, num_classes, queue_size)
    if validation_set and not validation_split:
      self.validation = Data('validation', test_videos_labels,
                             normalization, sequence_length,
                             crop_size, num_classes, queue_size)

  def get_videos_labels_lines(self, path):
    # Open the file according to the filename
    lines = open(path, 'r')
    lines = list(lines)
    return lines

  @property
  def data_shape(self):
    return (self._sequence_length, self._crop_size[1], self._crop_size[0], 3)

  @property
  def n_classes(self):
    return self._num_classes