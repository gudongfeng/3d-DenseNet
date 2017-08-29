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

  def normalize_image(self, image, normalization_type):
    """
    Args:
      image: numpy 3D array
      normalization_type: `str`, available choices:
        - divide_255
        - divide_256
        - std: (x - mean)/std
    """
    if normalization_type == 'divide_255':
      image = image/255
    elif normalization_type == 'divide_256':
      image = image/256
    elif normalization_type == 'std':
      image = (image - np.mean(image))/np.std(image)
    else:
      raise Exception("Unknow type of normalization")
    return image

  def labels_to_one_hot(self, labels, num_classes):
    """Convert 1D array of labels to one hot representation
    
    Args:
      labels: 1D numpy array
    """
    labels = labels
    new_labels = np.zeros((labels.shape[0], num_classes))
    new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
    return new_labels

  def labels_from_one_hot(self, labels):
    """Convert 2D array of labels to 1D class based representation
    
    Args:
      labels: 2D numpy array
    """
    return np.argmax(labels, axis=1)

class DataProvider:
  @property
  def data_shape(self):
    """Return shape as python list of one data entry"""
    raise NotImplementedError

  @property
  def n_classes(self):
    """Return `int` of num classes"""
    raise NotImplementedError
