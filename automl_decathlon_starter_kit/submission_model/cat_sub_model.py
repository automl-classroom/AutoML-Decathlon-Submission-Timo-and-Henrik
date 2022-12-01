import logging
import numpy as np
import sys
import time
import gc
import copy

import torch
from catboost import Pool, CatBoostClassifier, CatBoostRegressor

# seeding randomness for reproducibility
np.random.seed(42)

    
def get_cat_model(task_type:str, output_size: int, random_state=None):
    """
    ****************************************************************************
    ****************************************************************************
    Args:
        task_type: A string indicating the type of task to be solved.
        output_size: The number of labels.
        random_state: a random state for reproducability
    Returns:
        model: A configured CatBoost model for the given task.
    """
    # Common model params
    model_params = {
    'depth': 5, 
    'allow_const_label' : True
    }
    if random_state:
        model_params["random_seed"]=random_state
    iterations = max(15, (200 - output_size))
    print("Catboost dyn iterations: ", iterations)
    # Cases
    if task_type=="single-label":
        if output_size>2: # multi-class
            model_params = {
                **model_params,
                'task_type': 'GPU',
                'iterations': 1000,  
            }
        else:
            model_params = { # binary
                **model_params,
                'task_type': 'GPU',
                'iterations': 1000,  
            }
        model = CatBoostClassifier(**model_params)
    elif task_type=="multi-label":
        model_params = {
            **model_params,
            'loss_function': 'MultiLogloss',
                'task_type': 'CPU',
                'iterations': iterations,  
        }
        model = CatBoostClassifier(**model_params)
    elif task_type=="continuous":
        if output_size == 1: # single regression
            model_params = {
                **model_params,
                'task_type': 'GPU',
                'iterations': 1000,  
            }
        else:
            model_params = {
                **model_params,
                'loss_function': 'MultiRMSE',
                'task_type': 'CPU',
                'iterations': iterations,  
            }
        model = CatBoostRegressor(**model_params)
    else: 
        raise NotImplementedError
        
    return model

class Model:
    def __init__(self, metadata):
        '''
        The initalization procedure for your method given the metadata of the task
        '''
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        # Attribute necessary for ingestion program to stop evaluation process
        self.done_training = False
        self.metadata_ = metadata
        self.task = self.metadata_.get_dataset_name()
        self.task_type = self.metadata_.get_task_type()
        self.output_dim = np.prod(self.metadata_.get_output_shape())

        # Getting the device available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Device Found = ", self.device, "\nMoving Model and Data into the device..."
        )

        # Creating cat model
        self.model = get_cat_model(self.task_type, self.output_dim)
        
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
          remaining_time_budget: the time budget constraint for the task.
        """
        train_start = time.time()
        train_data = Pool(data = x,
                label = y)
        eval_data = Pool(data = x_val,
                label = y_val) 
        # Training (no loop)
        self.model.fit(
            train_data,
            eval_set=eval_data,
            use_best_model = True,
            verbose = 10,
        )
                
        train_end = time.time()


        train_duration = train_end - train_start
        self.total_train_time += train_duration
        logger.info(
            "{:.2f} sec used for catboost. ".format(
                train_duration
            )
            + "Total time used for training: {:.2f} sec. ".format(
                self.total_train_time
            )
        )
        gc.collect()


    def test(self, x_test, remaining_time_budget=None):
        """
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

        logger.info("Begin testing...")

        # get test predictions from the model

        if len(x_test.shape) > 2:
            x_test = copy.deepcopy(x_test)
            x_test = x_test.reshape((x_test.shape[0], -1))
        """
        
        if len(x_test.shape) > 2:
          x_test = x_test.tolist()
          for datapoint in x_test:
              datapoint.append(0)

        """

        predictions = self.model.predict(x_test)
        
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

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################


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
