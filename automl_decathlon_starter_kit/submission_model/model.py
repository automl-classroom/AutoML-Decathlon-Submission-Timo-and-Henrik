import setup
import logging
import numpy as np
import os
import sys
import time
import copy

#import models
from mlp_sub_model import Model as mlpmodel
from cat_sub_model import Model as catmodel
from xgb_sub_model import Model as xgbmodel
from layer_sub_model import Model as layer_model
from model_wrn_smac import Model as smac_model
from model_forecast import Model as forecast_model

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split


# seeding randomness for reproducibility
np.random.seed(42)
torch.manual_seed(1)


def merge_batches(dataloader: DataLoader, unnecessarily_inconsistent_data, sequence_size = 0):  
  """A data loader and preprocessor.
  ****************************************************************************
  ****************************************************************************
  Args:
    dataloader: a pytorch Dataloader. This dataloader can load training or test features and training labels.
    unnecessarily_inconsistent_data: a boolean indicating that for next prediction the loaded data would be under "data" instead of under "x" and "y"
    sequence_size: an Integer. For time series data this indicates the length of sequences.
  Returns:
    x_matrix: A `numpy.ndarray` matrix of shape (sample_count, input_dim). This array contains the training features for the stack ensemble.
    y_matrix: A `numpy.ndarray` matrix of shape (sample_count, output_dim). This array contains the training labels for the stack ensemble.
  """
  if unnecessarily_inconsistent_data:
    data = dataloader.dataset.dataset.data
    data_cutof = min(sequence_size, 50)
    lst = []
    #Fill and cut sequence length to a consistent length
    for i in data:
      hold = i[-data_cutof:]
      missing = max(0, (data_cutof - hold.shape[0]))
      if missing != 0:
        hold = np.pad(hold, ((missing, 0),(0, 0)), 'constant')

      lst.append(hold)
    data = np.asarray(lst)
    #Make features 1D
    data = np.reshape(data, (data.shape[0], -1))
    return data
  else:
    x_batches = []
    y_batches = []
    for x,y in dataloader:
        x = x.detach().numpy()
        x_batches.append(x)
        
        y = y.detach().numpy()
        y_batches.append(y)
    
    x_matrix = np.concatenate(x_batches, axis=0)

    data_cutof = min(sequence_size, 50)
    #Cut sequence length to a reasonable length for tabular ML Systems
    if sequence_size > 1:
      lst = []
      for i in x_matrix:
        hold = i[-data_cutof:]
        lst.append(hold)
      x_matrix = np.asarray(lst)
      #Make features 1D
      x_matrix = np.reshape(x_matrix, (x_matrix.shape[0], -1))
    y_matrix = np.concatenate(y_batches, axis=0)
    print("Data shapes: ", x_matrix.shape, y_matrix.shape)
    return x_matrix, y_matrix

def series_to_supervised(data, output_dim):
  """A data preprocessor to generate lables out of input data when the task is next prediction and no labels are given.
  ****************************************************************************
  ****************************************************************************
  Args:
    data: a `numpy.ndarray` matrix of shape (sample_count, output_dim * 50). This array should be two dimensional with the feature arrays containing up to fifty sets of data tuples in sequence.
    output_dim: an Integer. This is the size of one data tuple and thus the length of the output tuples for next prediction.
  Returns:
    x_ret: A `numpy.ndarray` matrix of shape (sample_count, output_dim * 49). This array contains the usable training features for the stack ensemble.
    y_ret: A `numpy.ndarray` matrix of shape (sample_count, output_dim). This array contains the usable training labels for the stack ensemble.
  """
  x_ret = []
  y_ret = []
  #Cut off the last data tuple to use for Labels
  """NOTE: This is most likely the issue that causes the stack ensemble to fail on the nottingham dataset because this step is not performed for test data, only for train and validation data.
  To fix this you would only have to reduce the sequence length for test data by one."""
  for datapoint in data:
    x = datapoint[:-output_dim]
    x_ret.append(x)

    y = datapoint[-output_dim:]
    y_ret.append(y)
  
  x_ret = np.array(x_ret)
  y_ret = np.array(y_ret)
  x_ret = np.squeeze(x_ret)
  y_ret = np.squeeze(y_ret)
  print("shape: ", y_ret[0], y_ret[1])
  return x_ret, y_ret


class Model:
    def __init__(self, metadata): 
        import xgboost as xgb
        print(xgb.__version__)
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        # Attribute necessary for ingestion program to stop evaluation process
        self.done_training = False
        
        #Metafeature Extraction
        self.metadata_ = metadata
        self.task = self.metadata_.get_dataset_name()
        self.task_type = self.metadata_.get_task_type()
        
        # Getting details of the data from meta data
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_dim = np.prod(self.metadata_.get_output_shape()) 
        self.input_dim = np.prod(self.metadata_.get_tensor_shape()[1:4])

        self.num_examples_train = self.metadata_.size()

        
        self.row_count, self.col_count = self.metadata_.get_tensor_shape()[2:4]
        self.channel = self.metadata_.get_tensor_shape()[1]
        self.sequence_size = self.metadata_.get_tensor_shape()[0]

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.total_test_time = 0
        
        self.train_batch_size = 64
        self.test_batch_size = 64



    def train(self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None):
        '''
        The training procedure of our method given training data, validation data, and remaining time budget for training.
        '''
        
        """Train this algorithm on the Pytorch dataset.
        ****************************************************************************
        ****************************************************************************
        Args:
          dataset: a `DecathlonDataset` object. Each of its examples is of the form
                (example, labels)
              where `example` is a dense 4-D Tensor of shape
                (sequence_size, row_count, col_count, num_channels)
              and `labels` is a 1-D or 2-D Tensor
          val_dataset: a 'DecathlonDataset' object. Is not 'None' if a pre-split validation set is provided, in which case you should use it for any validation purposes. Otherwise, you are free to create your own validation split(s) as desired.
          
          val_metadata: a 'DecathlonMetadata' object, corresponding to 'val_dataset'.
          remaining_time_budget: time remaining to execute train(). The method
              should keep track of its execution time to avoid exceeding its time
              budget. If remaining_time_budget is None, no time budget is imposed.
              
          remaining_time_budget: the time budget constraint for the task, which may influence the training procedure.
        """
        
        #############################
        #Analyze tensor shape and dataset shape to determine problem type
        #Possible types are: time_array, time_next, time_image, image_class, tab_class
        #############################
        space_dim = 0
        if self.col_count > 1:
          space_dim += 1
        if self.row_count > 1:
          space_dim += 1
        if self.channel > 1:
          space_dim += 1

        #Categorize the type of Problem based on the ammount of time and spacial dimensions of the data.
        self.weird_input = False
        if self.sequence_size > 1:  #Time series -> Forecasting
          if space_dim > 1:
            self.problem_type = 'time_image'
          elif hasattr(dataset.dataset, "data"):
            self.problem_type = 'time_next'
          else:
            self.problem_type = 'time_array'
        else:                       #Not Time series -> Classification/Regression
          if space_dim > 1:
            self.problem_type = 'image_class'
          else:
            self.problem_type = 'tab_class'

        if not hasattr(dataset.dataset, "x"):
          if self.problem_type == 'time_next':
            self.weird_input = True
          else:
            raise NotImplementedError
            
        
        # If PyTorch dataloader for training set doen't already exists, get the train dataloader
        if not hasattr(self, "trainloader"):
            self.trainloader = self.get_dataloader(
                dataset,
                self.train_batch_size,
                "train",
            )
        
        #############################
        #Load data
        #############################
        if not self.weird_input:
          x, y = merge_batches(self.trainloader, self.weird_input, self.sequence_size)
        else:
          data = merge_batches(self.trainloader, self.weird_input, self.sequence_size)
          x, y = series_to_supervised(data, self.output_dim)

        
        #############################
        #Load or create Validation Data
        #############################
        if val_dataset:
          valloader = self.get_dataloader(val_dataset, self.test_batch_size, "test")
          if not self.weird_input:
            x_valid, y_valid = merge_batches(valloader, self.weird_input, self.sequence_size)
          else:
            data = merge_batches(valloader, self.weird_input, self.sequence_size)
            x_valid, y_valid = series_to_supervised(data, self.output_dim)
        else:
          random_state=None # can set this for reproducibility if desired
          x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=random_state)


        training_metadata = copy.deepcopy(self.metadata_)
        training_metadata.set_size(x_train.shape[0])



        #############################
        #Create and train a stacked ensemble    time_array, time_next, time_image, image_class, tab_class
        #############################
        if self.problem_type in ['image_class']:
          self.model = smac_model(self.metadata_)
          self.model.train(dataset, val_dataset, val_metadata, remaining_time_budget)
        elif self.problem_type in ['time_image']:
          self.model = forecast_model(self.metadata_)
          self.model.train(dataset, val_dataset, val_metadata, remaining_time_budget)
        elif self.problem_type in ['time_array', 'tab_class','time_next']:

          #ensure each label has >1 classes since catboost has to be configured to accept the trainingdata otherwise and the MLP will not accept labels with only one class
          valid_metadata = copy.deepcopy(self.metadata_)
          valid_metadata.set_size(x_valid.shape[0])
          x_train = np.squeeze(x_train)
          y_train = np.squeeze(y_train)
          x_valid = np.squeeze(x_valid)
          y_valid = np.squeeze(y_valid)
          stack_shape = [catmodel, xgbmodel, mlpmodel]
          if self.task_type in["multi-label", "continuous"]:
            if len(y_train.shape) > 1:
              for i in range(y_train.shape[1]):
                if 1 >= len(np.unique(y_train[:, i])):
                  print("y col with less than 2 classes")
                  stack_shape = [xgbmodel, catmodel]
          self.model = layer_model(training_metadata, self.problem_type, models = stack_shape)
          self.model.train(x_train, y_train, x_valid, y_valid, remaining_time_budget)
        else:
          raise NotImplementedError

        




    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the Pytorch dataloader.
        Args:
          Same as that of `train` method, except that the `labels` will be empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
          remaining_time_budget: the remaining time budget left for testing, post-training 
        """
        test_begin = time.time()

        logger.info("Begin testing...")

        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )
        if self.problem_type in ['image_class']:
          predictions = self.model.test(dataset, remaining_time_budget)
        elif self.problem_type in ['time_image']:
          predictions = self.model.test(dataset, remaining_time_budget)
        elif self.problem_type in ['time_array', 'tab_class','time_next']:
          if self.weird_input:
            x_test = merge_batches(self.testloader, self.weird_input, self.sequence_size)
          else:
            x_test, _ = merge_batches(self.testloader, self.weird_input, self.sequence_size)

          x_test = np.squeeze(x_test)
          predictions = self.model.test(x_test)
        return predictions



    def get_dataloader(self, dataset, batch_size, split):
        """Get the PyTorch dataloader. Do not modify this method.
        Args:
          dataset:
          batch_size : batch_size for training set
        Return:
          dataloader: PyTorch Dataloader
        """
        if split == "train":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=dataset.collate_fn,
            )
        elif split == "test":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )
        return dataloader


def get_logger(verbosity_level):
    """Set logging format to something like:
    2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger("INFO")
