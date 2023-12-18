import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


class Scaler:
  def __init__(self, scale_down_factor=1_000_000):
    self.scale_down_factor = scale_down_factor
    self.minmaxscaler = MinMaxScaler()

  def fit(self, data):
    if not isinstance(data, np.ndarray):
      raise ValueError()

    assert len(data.shape) == 2

    data = data/self.scale_down_factor
    data = np.log(data)
    self.minmaxscaler.fit(data)

  def transform(self, data):
    if not isinstance(data, np.ndarray):
      raise ValueError()

    assert len(data.shape) == 2

    scaled_data = data/self.scale_down_factor
    scaled_data = np.log(scaled_data)
    scaled_data = self.minmaxscaler.transform(scaled_data)
    return scaled_data

  def invert(self, data):
    if not isinstance(data, np.ndarray):
      raise ValueError()

    assert len(data.shape) == 2

    inverted_data = self.minmaxscaler.inverse_transform(data)
    inverted_data = np.exp(inverted_data)
    inverted_data = inverted_data*self.scale_down_factor
    return inverted_data

def load_pretrained_scaler(pretrained_path):
  return torch.load(pretrained_path)