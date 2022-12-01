"""An example of code submission for the AutoML Decathlon challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test'). 

To create a valid submission, zip model.py and metadata together with other necessary files
such as tasks_to_run.yaml, Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.
"""

import datetime
import logging
import numpy as np
import os
import sys
import time
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from wrn1d import WideResNet1d
from wrn2d import WideResNet2d
from wrn3d import WideResNet3d

#from smac import BlackBoxFacade as BBFacade
#from smac import MultiFidelityFacade as MFFacade
import smac
from smac.scenario.scenario import Scenario
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_mf_facade import SMAC4MF

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

#from smac import Scenario
from ConfigSpace import ConfigurationSpace

from sklearn.model_selection import train_test_split
import copy

#sys.path.insert(1, '../scoring')
#sys.path.insert(1, 'scoring')
#from score import decathlon_scorer

from val_score_functions import decathlon_scorer

# seeding randomness for reproducibility
np.random.seed(42)
torch.manual_seed(1)

# Model class
class WideResNet(nn.Module):
    """
    Defines a module that will be created in '__init__' of the 'Model' class below, and will be used for training and predictions.
    """

    def __init__(self, input_shape, output_dim):
        super(WideResNet, self).__init__()

        fc_size = np.prod(input_shape)
        print("input_shape, fc_size", input_shape, fc_size)
        self.fc = nn.Linear(fc_size, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Model:
    def __init__(self, metadata):
        """
        The initalization procedure for your method given the metadata of the task
        """
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        # Attribute necessary for ingestion program to stop evaluation process
        self.metadata_ = metadata

        # Getting details of the data from meta data
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_dim = np.prod(self.metadata_.get_output_shape())

        self.num_examples_train = self.metadata_.size()

        row_count, col_count = self.metadata_.get_tensor_shape()[2:4]
        channel = self.metadata_.get_tensor_shape()[1]
        sequence_size = self.metadata_.get_tensor_shape()[0]

        self.num_train = self.metadata_.size()
        self.num_test = self.metadata_.get_output_shape()

        # Getting the device available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Device Found = ", self.device, "\nMoving Model and Data into the device..."
        )

        self.input_shape = (sequence_size, channel, row_count, col_count)
        self.sequence_size = sequence_size
        self.channel = channel
        self.row_count = row_count
        self.col_count = col_count
        self.task_type = self.metadata_.get_task_type()
        self.output_shape = self.metadata_.get_output_shape()
        print("\n\nINPUT SHAPE = ", self.input_shape)

        # PyTorch Optimizer and Criterion
        if self.metadata_.get_task_type() == "continuous":
            self.criterion = nn.MSELoss()
        elif self.metadata_.get_task_type() == "single-label":
            self.criterion = nn.CrossEntropyLoss()
        elif self.metadata_.get_task_type() == "multi-label":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.cumulated_num_steps = 0
        self.estimated_time_per_step = None
        self.total_test_time = 0
        self.estimated_time_test = None
        self.trained = False
        self.training_epochs = 20

        # no of examples at each step/batch
        self.train_batch_size = 128
        self.test_batch_size = 128

        self.best_score = np.inf

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

    def setup_model(self, cs):
        # getting an object for the PyTorch Model class for Model Class
        # use CUDA if available
        depth = 16
        spacetime_dims = np.count_nonzero(np.array(self.input_shape)[[0, 2, 3]] != 1)
        logger.info(f"Using WRN of dimension {spacetime_dims}")
        if spacetime_dims == 1:
            model = WideResNet1d(
                depth=depth,
                num_classes=self.output_dim,
                input_shape=self.input_shape,
                widen_factor=4,
                dropRate=cs["dropout"],
                in_channels=self.channel,
            )
        elif spacetime_dims == 2:
            model = WideResNet2d(
                depth=depth,
                num_classes=self.output_dim,
                input_shape=self.input_shape,
                widen_factor=4,
                dropRate=cs["dropout"],
                in_channels=self.channel,
            )
        elif spacetime_dims == 3:
            model = WideResNet3d(
                depth=depth,
                num_classes=self.output_dim,
                input_shape=self.input_shape,
                widen_factor=4,
                dropRate=cs["dropout"],
                in_channels=self.channel,
            )
        elif spacetime_dims == 0:  # Special case where we have channels only
            model = WideResNet1d(
                depth=depth,
                num_classes=self.output_dim,
                input_shape=self.input_shape,
                widen_factor=4,
                dropRate=cs["dropout"],
                in_channels=1,
            )
        else:
            raise NotImplementedError

        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cs["learning_rate"])

        return model, optimizer

    def train(
        self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None
    ):
        """
        CHANGE ME
        The training procedure of your method given training data, validation data (which is only directly provided in certain tasks, otherwise you are free to create your own validation strategies), and remaining time budget for training.
        """

        """Train this algorithm on the Pytorch dataset.

        This method will be called REPEATEDLY during the whole training/predicting
        process. So your `train` method should be able to handle repeated calls and
        hopefully improve your model performance after each call.

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
        """

        train_begin = time.time()
        
        self.complete_train_dataset = copy.deepcopy(dataset)

        if val_dataset == None:

            x_train, x_val, y_train, y_val = train_test_split(dataset.dataset.x, dataset.dataset.y)
        
            self.val_dataset = copy.deepcopy(dataset)
            self.val_dataset.dataset.x = x_val
            self.val_dataset.dataset.y = y_val

            self.dataset = dataset
            self.dataset.dataset.x = x_train
            self.dataset.dataset.y = y_train
        else:
            self.val_dataset = val_dataset

        # If PyTorch dataloader for training set doesn't already exist,
        # get the train dataloader
        if not hasattr(self, "trainloader"):
            self.complete_trainloader = self.get_dataloader(
                self.complete_train_dataset,
                self.train_batch_size,
                "train",
            )
            self.trainloader = DataLoader(
                self.dataset,
                self.train_batch_size
            )
            self.valloader = DataLoader(
                self.val_dataset,
                self.train_batch_size
            )

        # Training loop
        logger.info(f"epochs to train {self.training_epochs}")

        dropout = UniformFloatHyperparameter('dropout', 0.0, 0.9, default_value=0.7)
        learning_rate = UniformFloatHyperparameter('learning_rate', 1e-8, 1e-1, default_value=1e-3)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([dropout, learning_rate])
        scenario = Scenario({
        "run_obj": "quality",
        "wallclock-limit": remaining_time_budget*0.8,
        "cs": cs})

        #roughly estimation whether we deal with a large dataset / large samples or not
        if self.channel * self.row_count * self.col_count > 2500:
            max_epochs = 50
        else:
            max_epochs = 200

        intensifier_kwargs = {"initial_budget": 5, "max_budget": max_epochs, "eta": 3}

        smac = SMAC4MF(scenario=scenario,
               tae_runner=self.trainloop,
               intensifier_kwargs=intensifier_kwargs)

        tae = smac.get_tae_runner()

        self.cur_time = time.time()
        self.remaining_time_budget = remaining_time_budget
        self.first_max_epoch_time = None

        def_value = tae.run(config=cs.get_default_configuration(), budget=max_epochs, seed=0)

        self.first_max_epoch_time = time.time() - train_begin
        self.remaining_time_budget -= self.first_max_epoch_time
        self.time_checkpoint = time.time()

        try:
            incumbent = smac.optimize()
        except:
            print("Abort to prevent timeout")
            pass
        finally:
            pass
        print(self.best_score)
        
    def validation(self, dataloader, model):
        preds = []
        with torch.no_grad():
            model.eval()
            for x, _ in iter(dataloader):
                if torch.cuda.is_available():
                    x = x.float().cuda()
                else:
                    x = x.float()
                logits = model(x)

                # Choose correct prediction type
                if self.metadata_.get_task_type() == "continuous":
                    pred = logits
                elif self.metadata_.get_task_type() == "single-label":
                    pred = torch.softmax(logits, dim=1).data
                elif self.metadata_.get_task_type() == "multi-label":
                    pred = torch.sigmoid(logits).data
                else:
                    raise NotImplementedError

                preds.append(pred.cpu().numpy())

        preds = np.vstack(preds)
        return preds

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the Pytorch dataloader.

        Args:
          Same as that of `train` method, except that the `labels` will be empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
        """
        test_begin = time.time()

        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )

        # get predictions from the test loop
        predictions = self.testloop(self.testloader)

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

    ############################################################################
    ### Above 3 methods (__init__, train, test) should always be implemented ###
    ############################################################################

    def trainloop(self, cs, budget):
        """Training loop with no of given steps
        Args:
          criterion: PyTorch Loss function
          Optimizer: PyTorch optimizer for training
          steps: No of steps to train the model

        Return:
          None, updates the model parameters
        """
        
        

        model, optimizer = self.setup_model(cs)

        model.train()
        for _ in tqdm(range(int(budget)), desc="Epochs trained", position=0):
            if self.first_max_epoch_time is not None:
                self.time_since_last = time.time() - self.time_checkpoint
                self.time_checkpoint = time.time()
                self.remaining_time_budget -= self.time_since_last
                self.possible_runs = self.remaining_time_budget // self.first_max_epoch_time - 1
                if self.possible_runs <= 0:
                    return smac.tae.TAEAbortException()
            for x, y in tqdm(
                self.trainloader, desc="Batches this epoch", position=1, leave=False, disable=True
            ):
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                optimizer.zero_grad()

                logits = model(x)
                loss = self.criterion(logits, y.reshape(y.shape[0], -1))

                if hasattr(self, "scheduler"):
                    self.scheduler.step(loss)

                loss.backward()
                optimizer.step()
        model.eval()
        solution = self.val_dataset.dataset.y
        preds = self.validation(self.valloader, model)
        score = decathlon_scorer(solution, preds, self.sequence_size, self.channel,
                                self.row_count, self.col_count, self.output_shape, self.task_type)
        if score == False:
            with torch.no_grad():
                score = self.criterion(torch.tensor(preds), torch.tensor(solution)).item()
        if score < self.best_score:
            self.best_score = score
            self.model = model
            self.optimizer = optimizer
        print(score)

        return score

    def testloop(self, dataloader):
        """
        Args:
          dataloader: PyTorch test dataloader

        Return:
          preds: Predictions of the model as Numpy Array.
        """
        preds = []
        with torch.no_grad():
            self.model.eval()
            for x, _ in iter(dataloader):
                if torch.cuda.is_available():
                    x = x.float().cuda()
                else:
                    x = x.float()
                logits = self.model(x)

                # Choose correct prediction type
                if self.metadata_.get_task_type() == "continuous":
                    pred = logits
                elif self.metadata_.get_task_type() == "single-label":
                    pred = torch.softmax(logits, dim=1).data
                elif self.metadata_.get_task_type() == "multi-label":
                    pred = torch.sigmoid(logits).data
                else:
                    raise NotImplementedError

                preds.append(pred.cpu().numpy())

        preds = np.vstack(preds)
        return preds


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
