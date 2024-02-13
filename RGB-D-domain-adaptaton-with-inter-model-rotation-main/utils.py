import functools
from collections import namedtuple
from itertools import product
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from typing import Sequence, Text, Union
import torch.nn as nn
import os
import torch.optim as opt
import pandas as pd


# Used both for evaluation and testing 
class EvaluationManager:
    def __init__(self, nets):
        self.nets = nets

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        for net in self.nets:
            net.eval()

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        for net in self.nets:
            net.train()
        return False

    def __call__(self, func): 
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad

# Iterator over a list of parameters
class RunBuilder():
  @staticmethod
  def get_runs(params):
    Run = namedtuple('Run', params.keys())
    runs=[]
    for v in product(*params.values()):
       runs.append(Run(*v))
    return runs

def listToString(params):
  hp_string = str(params.lr) + str(params.batch_size)
  return hp_string


def entropy_loss_paper(logits):
    p_softmax = F.softmax(logits, dim=1) #this returns a tensor
    mask = p_softmax.ge(0.000001)  # greater or equal to threshold value
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))

# This makes an object as an iterable
class DataWraper:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def __iter__(self):
        self.iterator = iter(self.loader)

    def get_next(self):
        try:
            items = self.iterator.next()
        except:
            self.__iter__()
            items = self.iterator.next()
        return items


class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims  # if isinstance(optims, Iterable) else [optims]

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()   # zeroing the gradient to avoid accumulation of gradient

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True

# A class tha creates a DataFrame with all the risults of each experiment.
class RunRecord():
    def __init__(self):
      self.run_record = []    
      self.run_parameters = None
      self.epoch = 0
      self.df = None


      self.classification_loss = 0.0
      self.classification_accuracy =0.0

      self.source_rot_loss = 0.0
      self.source_rot_accuracy = 0.0

      self.test_rot_loss = 0.0
      self.test_rot_accuracy = 0.0

    def start_epoch(self, epoch, parameters):
        self.epoch = epoch
        self.run_parameters = parameters


    def update_classification(self, loss, accuracy):
        self.classification_accuracy = accuracy
        self.classification_loss = loss


    def update_source_rot(self, loss, accuracy):
        self.source_rot_accuracy = accuracy
        self.source_rot_loss = loss


    def update_test_rot(self, loss, accuracy):
        self.test_rot_accuracy = accuracy
        self.test_rot_loss = loss

    
    def end_epoch(self):
        results = dict()
        results["epoch_running"] = self.epoch
        results["classification_loss"] = self.classification_loss
        results["classification_accuracy"] = self.classification_accuracy

        results["source_rot_accuracy"] = self.source_rot_accuracy
        results["source_rot_loss"] = self.source_rot_loss

        results["test_rot_accuracy"] = self.test_rot_accuracy
        results["test_rot_loss"] = self.test_rot_loss


        for k,v in self.run_parameters._asdict().items():  results[k] = v

        self.run_record.append(results)
        df = pd.DataFrame.from_dict(self.run_record, orient='columns')
        self.df = df
        print("results")
        print(results)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum()

def save_checkpoint(path: Text,
                    epoch: int,
                    modules: Union[nn.Module, Sequence[nn.Module]],
                    optimizers: Union[opt.Optimizer, Sequence[opt.Optimizer]],
                    safe_replacement: bool = True):
    """
    Save a checkpoint of the current state of the training, so it can be resumed.
    This checkpointing function assumes that there are no learning rate schedulers or gradient scalers for automatic
    mixed precision.
    :param path:
        Path for your checkpoint file
    :param epoch:
        Current (completed) epoch
    :param modules:
        nn.Module containing the model or a list of nn.Module objects
    :param optimizers:
        Optimizer or list of optimizers
    :param safe_replacement:
        Keep old checkpoint until the new one has been completed
    :return:
    """

    # This function can be called both as
    # save_checkpoint('/my/checkpoint/path.pth', my_epoch, my_module, my_opt)
    # or
    # save_checkpoint('/my/checkpoint/path.pth', my_epoch, [my_module1, my_module2], [my_opt1, my_opt2])
    if isinstance(modules, nn.Module):
        modules = [modules]
        
    if isinstance(optimizers, opt.Optimizer):
        optimizers = [optimizers]

    # Data dictionary to be saved
    data = {
        'epoch': epoch,
        # State dict for all the modules
        'modules': [m.state_dict() for m in modules],
        # State dict for all the optimizers
        'optimizers': [o.state_dict() for o in optimizers]
    }

    # Safe replacement of old checkpoint
    temp_file = None
    if os.path.exists(path) and safe_replacement:
        # There's an old checkpoint. Rename it!
        temp_file = path + '.old'
        os.rename(path, temp_file)

    # Save the new checkpoint
    with open(path, 'wb') as fp:
        torch.save(data, fp)
        # Flush and sync the FS
        fp.flush()
        os.fsync(fp.fileno())

    # Remove the old checkpoint
    if temp_file is not None:
        os.unlink(path + '.old')

def load_checkpoint(path: Text,
                    default_epoch: int,
                    modules: Union[nn.Module, Sequence[nn.Module]],
                    optimizers: Union[opt.Optimizer, Sequence[opt.Optimizer]],
                    verbose: bool = True):
    """
    Try to load a checkpoint to resume the training.
    :param path:
        Path for your checkpoint file
    :param default_epoch:
        Initial value for "epoch" (in case there are not snapshots)
    :param modules:
        nn.Module containing the model or a list of nn.Module objects. They are assumed to stay on the same device
    :param optimizers:
        Optimizer or list of optimizers
    :param verbose:
        Verbose mode
    :return:
        Next epoch
    """
    if isinstance(modules, nn.Module):
        modules = [modules]
    if isinstance(optimizers, opt.Optimizer):
        optimizers = [optimizers]

    # If there's a checkpoint
    if os.path.exists(path):
        # Load data
        data = torch.load(path, map_location=next(modules[0].parameters()).device)

        # Load state for all the modules
        for i, m in enumerate(modules):

            modules[i].load_state_dict(data['modules'][i])

        # Load state for all the optimizers
        for i, o in enumerate(optimizers):
            optimizers[i].load_state_dict(data['optimizers'][i])

        # Next epoch
        return data['epoch'] + 1
    else:
        return default_epoch