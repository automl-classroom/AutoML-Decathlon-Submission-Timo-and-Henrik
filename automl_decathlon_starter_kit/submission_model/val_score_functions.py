from functools import partial
from sklearn.metrics import (
    zero_one_loss,
    roc_auc_score,
    f1_score,
    average_precision_score,
)
import torch
import numpy as np

#Since we do not have access to the original decathlon scorer function, we decided to build our own
#We assume that tasks that are part of the secret tasks will behave similarly to the evaluation task
#Because of that we use the same validation metrics to determine the best hyperparameter configuration
#In case we encounter an task of unknown format, we fall back to standard validation loss
def decathlon_scorer(solution, prediction, sequence_size,
    channel, row_count, col_count, output_shape, task_type):
    if len(output_shape) > 1:
        solution = solution.reshape(-1)
        prediction = prediction.reshape(-1)
    try:
        if channel > 1 and row_count > 1 and task_type == "single-label":
            score = zero_one_error(solution, prediction)
        elif channel > 1 and row_count > 1 and task_type == "multi-label":
            score = inv_map_score(solution, prediction)
        elif row_count > 1 and col_count and task_type == "multi-label":
            score = inv_auroc_score(solution, prediction)
        else:
            #score = inv_auroc_score(solution, prediction)
            score = False
    except:
        score = False
    return score
'''
def decathlon_scorer(solution, prediction, sequence_size,
    channel, row_count, col_count, output_shape, task_type):
    space_dimensions = 0
    if channel > 1:
        space_dimensions += 1
    if row_count > 1:
        space_dimensions += 1
    if col_count > 1:
        space_dimensions += 1
    if sequence_size > 1:
        if (output_shape == [row_count, col_count] or \
           output_shape == [col_count, row_count] or \
           output_shape == [row_count, channel] or \
           output_shape == [channel, row_count] or \
           output_shape == [channel, col_count] or \
           output_shape == [col_count, channel]):
              if space_dimensions >= 2:
                  if task_type == "continuous":
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = l2_relative_error(solution, prediction)
                  elif task_type == "single-label":
                      #default scorer
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = l2_relative_error(solution, prediction)
                  elif task_type == "multi-label":
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = inv_auroc_score(solution, prediction)
              else:
                  if task_type == "continuous":
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = l2_relative_error(solution, prediction)
                  elif task_type == "single-label":
                      #default scorer
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = l2_relative_error(solution, prediction)
                  elif task_type == "multi-label":
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = inv_auroc_score(solution, prediction)
        else:
            if task_type == "continuous":
                solution = solution.reshape(-1)
                prediction = prediction.reshape(-1)
                score = l2_relative_error(solution, prediction)
            elif task_type == "single-label":
                #default scorer
                solution = solution.reshape(-1)
                prediction = prediction.reshape(-1)
                score = l2_relative_error(solution, prediction)
            elif task_type == "multi-label":
                solution = solution.reshape(-1)
                prediction = prediction.reshape(-1)
                score = inv_auroc_score(solution, prediction)
    else:
      if (output_shape == [row_count, col_count] or \
           output_shape == [col_count, row_count] or \
           output_shape == [row_count, channel] or \
           output_shape == [channel, row_count] or \
           output_shape == [channel, col_count] or \
           output_shape == [col_count, channel]):
              if space_dimensions >= 2:
                  if task_type == "continuous":
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = l2_relative_error(solution, prediction)
                  elif task_type == "single-label":
                      #default scorer
                      score = inv_f1_score(solution, prediction)
                  elif task_type == "multi-label":
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = inv_auroc_score(solution, prediction)
              else:
                  if task_type == "continuous":
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = l2_relative_error(solution, prediction)
                  elif task_type == "single-label":
                      #default scorer
                      score = inv_f1_score(solution, prediction)
                  elif task_type == "multi-label":
                      solution = solution.reshape(-1)
                      prediction = prediction.reshape(-1)
                      score = inv_auroc_score(solution, prediction)
      else:
            if task_type == "continuous":
                score = l2_relative_error(solution, prediction)
            elif task_type == "single-label":
                print(solution.shape, prediction.shape)
                score = inv_f1_score(solution, prediction)
            elif task_type == "multi-label":
                score = inv_auroc_score(solution, prediction)
    return score
'''

class RelativeError(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(RelativeError, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def __call__(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms


def loss_metric(solution, prediction, loss_class):
    loss = loss_class()
    return loss(torch.Tensor(solution), torch.Tensor(prediction)).data.item()


def l2_relative_error(solution, prediction, size_average=True):
    return loss_metric(
        solution, prediction, partial(RelativeError, size_average=size_average)
    )


def zero_one_error(solution, prediction):
    y = np.argmax(solution, axis=1)
    y_hat = np.argmax(prediction, axis=1)
    return zero_one_loss(y, y_hat)


def inv_f1_score(solution, prediction):
    y = np.argmax(solution, axis=1)
    y_hat = np.argmax(prediction, axis=1)
    return 1 - f1_score(y, y_hat, average="macro")


def inv_auroc_score(solution, prediction):
    y = solution > 0.5
    y_hat = prediction
    return 1.0 - roc_auc_score(y, y_hat, average="macro")


def inv_map_score(solution, prediction):
    y = solution
    y_hat = prediction
    return 1.0 - average_precision_score(y, y_hat, average="micro")


def nll_score(solution, prediction):
    return loss_metric(solution, prediction, partial(torch.nn.BCELoss))


def false_negative_rate(solution, prediction):
    solution = solution.reshape(solution.shape[0], -1)
    prediction = prediction.reshape(prediction.shape[0], -1)
    y = solution > 0.5
    y_hat = prediction > 0.5
    if len(y_hat.shape) == 2:
        y_hat = y_hat.reshape(1, *y_hat.shape)
    if len(y.shape) == 2:
        y = y.reshape(1, *y.shape)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(y.shape[0]):
        TP += (y_hat[i][y[i] == 1] == 1).sum()
        TN += (y_hat[i][y[i] == 0] == 0).sum()
        FP += (y_hat[i][y[i] == 0] == 1).sum()
        FN += (y_hat[i][y[i] == 1] == 0).sum()
    return FN / (FN + TP)
