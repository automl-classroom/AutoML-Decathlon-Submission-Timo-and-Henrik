import datetime
import logging
import numpy as np
import os
import sys
import time
import math
import copy
from collections import Counter


#from model import model
from mlp_sub_model import Model as mlpmodel
from cat_sub_model import Model as catmodel
from xgb_sub_model import Model as xgbmodel

import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, r2_score

# seeding randomness for reproducibility
np.random.seed(42)
torch.manual_seed(1)



def one_hot_decode(y, y_val):
  """A data preprocessor decoding one hot encoded labels.
  ****************************************************************************
  ****************************************************************************
  Args:
    y: A `numpy.ndarray` matrix of shape (sample_count, output dim).
    y_val: A `numpy.ndarray` matrix of shape (validation_sample_count, output dim).
  Returns:
    y: A `numpy.ndarray` array of length (sample_count).
    y_val: A `numpy.ndarray` array of length (sample_count).
  """
  y_ind = []
  y_val_ind = []
  for i in y:
    y_ind.append(np.argmax(i, axis=0))
  for i in y_val:
    y_val_ind.append(np.argmax(i, axis=0))
  return np.asarray(y_ind), np.asarray(y_val_ind)

def ohe(predictions):
  """A data preprocessor encoded one hot decoding labels.
  ****************************************************************************
  ****************************************************************************
  Args:
    predictions: A `numpy.ndarray` array of length (sample_count).
  Returns:
    predictions: A `numpy.ndarray` matrix of shape (sample_count, output dim).
  """
  ohe = np.zeros((predictions.size, predictions.max() + 1))
  ohe[np.arange(predictions.size), predictions] = 1
  return ohe


class Model:
    def __init__(self, metadata, problem, models = [xgbmodel, catmodel, mlpmodel], num_levels = 2):
        '''
        The initalization procedure for your method given the metadata of the task
        '''
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
          problem: a string. This indicates what type of task out of 'time_array', 'tab_class' and 'time_next' the dataset contains
          models: a list of ml models. ml models must be implemented as the three default choices are.
          num_levels: an Integer. It gives the maximum number of layers in the stack ensemble
        """
        #########################
        #Metafeature Extraction
        #########################
        # Attribute necessary for ingestion program to stop evaluation process
        self.done_training = False
        self.metadata_ = metadata
        self.task = self.metadata_.get_dataset_name()
        self.task_type = self.metadata_.get_task_type()
        self.problem_type = problem
        
        # Getting details of the data from meta data
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_dim = np.prod(self.metadata_.get_output_shape()) 
        self.input_dim = np.prod(self.metadata_.get_tensor_shape()) 

        self.num_examples_train = self.metadata_.size()

        self.num_levels = num_levels
        self.num_tasks = len(models) * num_levels


        #Lower level metadata
        stacked_metadata = copy.deepcopy(metadata)
        if self.task_type == 'single-label':
          appended_data_length = len(models)
        else:
          appended_data_length = self.output_dim * len(models)
        stacked_metadata.metadata_["input_shape"] = (1, (self.input_dim + appended_data_length), 1, 1)


        ############################
        # Creating a Stack of Models
        ############################
        self.model = []
        layer = []
        for model in models:
          layer.append(model(metadata))
        self.model.append(layer)
        for i in range(1, num_levels):
          layer = []
          for model in models:
            layer.append(model(stacked_metadata))
          self.model.append(layer)
        
        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.total_test_time = 0



    def train(self, x, y, x_val, y_val, remaining_time_budget=None):
        """
        ****************************************************************************
        ****************************************************************************
        Args:
          x: A `numpy.ndarray` matrix of shape (sample_count, input_dim). It contains features of the training data
          y: A `numpy.ndarray` matrix of shape (sample_count, output_dim). It contains labeös of the training data
          x_val: A `numpy.ndarray` matrix of shape (sample_count, input_dim). It contains features of the training data
          y_val: A `numpy.ndarray` matrix of shape (sample_count, output_dim). It contains labeös of the training data
          remaining_time_budget: the time budget constraint for the task, which limits the training procedure should it appear after one layer to exeed the available time.
        """

        train_start = time.time()
        
        if remaining_time_budget:
          time_per_task = remaining_time_budget / self.num_tasks
        else:
          time_per_task = None


        if len(x.shape) > 2:
            x = copy.deepcopy(x)
            x = x.reshape((x.shape[0], -1))
            x_val = copy.deepcopy(x_val)
            x_val = x_val.reshape((x_val.shape[0], -1))

        #Decode one hot encoding for single label
        if self.task_type == 'single-label':
          y, y_val = one_hot_decode(y, y_val)
        
        #Prepare data for deeper layers and reshape data
        x_layer = copy.deepcopy(x)
        x_layer = x_layer.reshape((x.shape[0], -1))
        x_layer_val = copy.deepcopy(x_val)
        x_layer_val = x_layer_val.reshape((x_val.shape[0], -1))

        x_next_layer = x_layer
        x_next_layer_val = x_layer_val

        if len(x.shape) > 3:
          x = x.reshape((x.shape[0], -1, np.prod(x.shape[2:])))
          x_val = x_val.reshape((x_val.shape[0], -1, np.prod(x_val.shape[2:])))

        self.top_model = (0, 0)
        self.top_acc = -100

        # Train first layer
        for model in self.model[0]:
          print("Training model:", model)
          model_accuracy = 0
          model.train(x, y, x_val, y_val, time_per_task)
          print("Testing model:", model)
          model_prediction =  model.test(x)
          val_prediction = model.test(x_val)
          print("individual model shape", val_prediction.shape)
          # Evaluate performance of current model
          if len(y_val.shape) > 1:
            for i in range(0, val_prediction.shape[1]):
              if self.task_type == "continuous":
                model_accuracy += r2_score(val_prediction[:,i], y_val[:,i])
              else:
                model_accuracy += balanced_accuracy_score(val_prediction[:,i], y_val[:,i])
            model_accuracy /= y_val.shape[1]
          else:
            if self.task_type == "continuous":
              model_accuracy += r2_score(val_prediction, y_val)
            else:
              model_accuracy = balanced_accuracy_score(val_prediction, y_val)
          if model_accuracy > self.top_acc:
            self.top_acc = model_accuracy
            self.top_model = (0, self.model[0].index(model))
          # Model predictions must be 2D Array
          if len(model_prediction.shape) == 1:
            model_prediction.resize(model_prediction.shape[0], 1)
            val_prediction.resize(val_prediction.shape[0], 1)
          x_next_layer = np.append(x_next_layer, model_prediction, 1)
          x_next_layer_val = np.append(x_next_layer_val, val_prediction, 1)


        train_end = time.time()
        train_duration = train_end - train_start

        # Adjust layers for time constraints
        self.num_levels = min (self.num_levels, int(remaining_time_budget/train_duration))


        # Train other layers
        for i in range(1, self.num_levels):
          x_predictions, x_val_predictions = self.train_layer(i, x_next_layer, y, x_next_layer_val, y_val, time_per_task)
          x_predictions = np.array(x_predictions)
          x_predictions = np.transpose(x_predictions, (1, 0, 2))
          x_predictions = np.reshape(x_predictions, (x_predictions.shape[0], -1))

          
          x_val_predictions = np.array(x_val_predictions)
          x_val_predictions = np.transpose(x_val_predictions, (1, 0, 2))
          x_val_predictions = np.reshape(x_val_predictions, (x_val_predictions.shape[0], -1))

          print("individual model shape", x_val_predictions.shape)
          x_next_layer = np.append(x_layer, x_predictions, 1)
          x_next_layer_val = np.append(x_layer_val, x_val_predictions, 1)
                  


        train_end = time.time()


        train_duration = train_end - train_start
        self.total_train_time += train_duration
        logger.info(
            "{:.2f} sec used for stack. ".format(
                train_duration
            )
            + "Total time used for training: {:.2f} sec. ".format(
                self.total_train_time
            )
        )

    def train_layer(self, layer, x, y, x_val, y_val, time_per_task = None):
        """
        ****************************************************************************
        ****************************************************************************
        Args:
          layer: An Integer. It indicates the layer currently being trained for array access.
          x: A `numpy.ndarray` matrix of shape (sample_count, input_dim). It contains features of the training data
          y: A `numpy.ndarray` matrix of shape (sample_count, output_dim). It contains labeös of the training data
          x_val: A `numpy.ndarray` matrix of shape (sample_count, input_dim). It contains features of the training data
          y_val: A `numpy.ndarray` matrix of shape (sample_count, output_dim). It contains labeös of the training data
          time_per_task: the time budget constraint for the task.
        Returns:
          x_predictions: A `numpy.ndarray` array of length (sample_count). It containes the predictions of the current layer for the training data.
          x_val_predictions: A `numpy.ndarray` array of length (sample_count).It containes the predictions of the current layer for the validation data.
        """
        x_predictions = []
        x_val_predictions = []
        for model in self.model[layer]:
          print("Training model:", model)
          model.train(x, y, x_val, y_val, time_per_task)
          print("Testing model:", model)
          model_prediction =  model.test(x)
          val_prediction = model.test(x_val)
          
          if len(model_prediction.shape) == 1:
            model_prediction.resize(model_prediction.shape[0], 1)
            val_prediction.resize(val_prediction.shape[0], 1)
          x_predictions.append(model_prediction)
          x_val_predictions.append(val_prediction)


        #Evaluate the entire layer
        model_accuracy = 0
        predictions = self.voting(x_val_predictions)
        if len(y_val.shape) > 1:
          for i in range(0, predictions.shape[1]):
            if self.task_type == "continuous":
              model_accuracy += r2_score(val_prediction[:,i], y_val[:,i])
            else:
              model_accuracy += balanced_accuracy_score(val_prediction[:,i], y_val[:,i])
          model_accuracy /= y_val.shape[1]
        else:
          if self.task_type == "continuous":
            model_accuracy += r2_score(val_prediction, y_val)
          else:
            model_accuracy = balanced_accuracy_score(val_prediction, y_val)
        if model_accuracy > self.top_acc:
          self.top_acc = model_accuracy
          self.top_model = (layer, -1)

        
        
        return x_predictions, x_val_predictions


    def test_layer(self, layer, dataset,  time_per_task = None):
        """
        ****************************************************************************
        ****************************************************************************
        Args:
          layer: An Integer. It indicates the layer currently being trained for array access.
          x: A `numpy.ndarray` matrix of shape (sample_count, input_dim). It contains features of the training data
          time_per_task: the time budget constraint for the task.
        Returns:
          predictions: A `numpy.ndarray` array of length (sample_count). It containes the predictions of the current layer for the training data.
        """
        predictions = []
        for model in self.model[layer]:
          predictions.append(model.test(dataset))
        return predictions



    def test(self, x_test, remaining_time_budget=None):
        """Test this algorithm on the numpy array containing the test data.
        Args:
          x: A `numpy.ndarray` matrix of shape (sample_count, input_dim). It contains features of the training data
          remaining_time_budget: the time budget constraint for the task.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
        """

        test_begin = time.time()

        #Safety reshape of training data
        if len(x_test.shape) > 2:
            x_test = copy.deepcopy(x_test)
            x_test = x_test.reshape((x_test.shape[0], -1))

        if remaining_time_budget:
          time_per_task = remaining_time_budget / self.num_tasks
        else:
          time_per_task = None

        logger.info("Begin testing...")

        if self.top_model[0] == 0:#single model outperforms stack
          model = self.model[0][self.top_model[1]]
          predictions = model.test(x_test)

        else: #use stack
          x_layer = copy.deepcopy(x_test)
          x_layer = x_layer.reshape((x_test.shape[0], -1))
          x_next_layer = x_layer
          # Get predictions of the first layer
          for model in self.model[0]:
            model_prediction = model.test(x_test)

            if len(model_prediction.shape) == 1:
              model_prediction.resize(model_prediction.shape[0], 1)
            x_next_layer = np.append(x_next_layer, model_prediction, 1)


          # Get predictions of the other layers
          for i in range(1, self.num_levels):
            x_predictions = self.test_layer(i, x_next_layer, time_per_task)

            x_predictions = np.array(x_predictions)
            predictions = copy.deepcopy(x_predictions)
            x_predictions = np.reshape(x_predictions, (x_predictions.shape[0], x_predictions.shape[1], -1))
            x_predictions = np.transpose(x_predictions, (1, 0, 2))
            x_predictions = np.reshape(x_predictions, (x_predictions.shape[0], -1))

            x_next_layer = np.append(x_layer, x_predictions, 1)

          predictions = self.voting(predictions)

        print("individual model shape", predictions.shape)
        if self.task_type == 'single-label':
          predictions = ohe(predictions)
        
        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration

        logger.info(
            "[+] Successfully made one prediction. {:.2f} sec used. ".format(
                test_duration
            )
            + "Total time used for testing: {:.2f} sec. ".format(self.total_test_time)
        )
        return predictions

    def voting(self, list_of_pred):
        """
        ****************************************************************************
        ****************************************************************************
        Args:
          list_of_pred: A `numpy.ndarray` matrix of shape (sample_count, number of models, input_dim). It contains predictions of the individual models
        Returns:
          result: A `numpy.ndarray` array of length (sample_count). It containes the predictions of the current layer.
        """
        array_predictions = np.array(list_of_pred)
        array_predictions = np.squeeze(array_predictions)
        if self.task_type == 'continuous':
          # Averaging for continuous data
          result = np.zeros(array_predictions[0].shape, dtype = float)
          for vote in array_predictions:
            result += vote
          result /= len(array_predictions) 
        else:
          # Voting for classification
          result = np.zeros(array_predictions[0].shape, dtype = int)
          for i in range(0, array_predictions.shape[1]):
            if self.task_type == 'single-label':
              line_predictions = array_predictions[:, i]
              counts = Counter(line_predictions)
              result[i] = counts.most_common(1)[0][0]
            elif self.task_type == 'multi-label':
              for j in range(0, array_predictions.shape[2]):###
                line_predictions = array_predictions[:, i, j]
                counts = Counter(line_predictions)
                result[i][j] = counts.most_common(1)[0][0]
        return result





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
