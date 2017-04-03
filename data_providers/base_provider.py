import numpy as np


class DataSet:
  """Class to represent some dataset: train, validation, test"""
  @property
  def num_examples(self):
    """Return qtty of examples in dataset"""
    raise NotImplementedError

  def next_batch(self, batch_size):
    """Return batch of required size of data, labels"""
    raise NotImplementedError


class VideosDataset(DataSet):
  """Dataset for videos that provide some often used methods"""

  def _measure_mean_and_std(self):
    means = []
    stds = []
    # For each image in the videos (assume the second dimension)
    # self.videos shape:
    #   [num_examples, sequence_length, height, width, channel]
    # For every channel in image (assume this is the last dimension)
    for ch in range(self.videos.shape[-1]):
      means.append(np.mean(self.images[:, :, :, :, ch]))
      stds.append(np.std(self.videos[:, :, :, :, ch]))
    self._means = means
    self._stds = stds

  @property
  def videos_means(self):
    if not hasattr(self, '_means'):
      self._measure_mean_and_std()
    return self._means

  @property
  def videos_std(self):
    if not hasattr(self, '_stds'):
      self._measure_mean_and_std()
    return self._stds

  def shuffle_videos_and_labels(self, videos, labels):
    rand_indexes = np.random.permutation(videos.shape[0])
    shuffled_images = images[rand_indexes]
    shuffled_labels = labels[rand_indexes]
    return shuffled_images, shuffled_labels

  def normalize_videos(self, videos, normalization_type):
    """
    Args:
      videos: numpy 5D array
      normalization_type: `str`, available choices:
        - divide_255
        - divide_256
        - by_channels
    """
    if normalization_type == 'divide_255':
      videos = videos/255
    elif normalization_type == 'divide_256':
      videos = videos/256
    elif normalization_type == 'by_channels':
      videos = videos.astype('float64')
      # for every channel in image(assume this is last dimension)
      for i in range(videos.shape[-1]):
        videos[:, :, :, :, i] = ((videos[:, :, :, :, i] - self.videos_means[i]) / self.videos_std[i])
    else:
      raise Exception("Unknow type of normalization")
    return videos

  def normalize_all_videos_by_channels(self, initial_videos):
    new_videos = np.zeros(initial_videos.shape)
    for i in range(initial_videos.shape[0]):
      for j in range(initial_videos.shape[1]):
        new_videos[i][j] = self.normalize_image_by_channel(initial_videos[i][j])
    return new_videos

  def normalize_image_by_channel(self, image):
    new_image = np.zeros(image.shape)
    for chanel in range(3):
      mean = np.mean(image[:, :, chanel])
      std = np.std(image[:, :, chanel])
      new_image[:, :, chanel] = (image[:, :, chanel] - mean) / std
    return new_image


class DataProvider:
  @property
  def data_shape(self):
    """Return shape as python list of one data entry"""
    raise NotImplementedError

  @property
  def n_classes(self):
    """Return `int` of num classes"""
    raise NotImplementedError

  def labels_to_one_hot(self, labels):
    """Convert 1D array of labels to one hot representation
    
    Args:
      labels: 1D numpy array
    """
    labels = np.array(labels)
    new_labels = np.zeros((labels.shape[0], self.n_classes))
    new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
    return new_labels

  def labels_from_one_hot(self, labels):
    """Convert 2D array of labels to 1D class based representation
    
    Args:
      labels: 2D numpy array
    """
    return np.argmax(labels, axis=1)
