from bilmo.scripts.config import Config
from fastai.torch_core import add_metrics, listify
from fastai.basic_train import LearnerCallback, Learner
import torch
from functools import partial
import logging
conf = Config.conf
log = logging.getLogger("cafa-logger")

class CafaAssesment(LearnerCallback):
    "Computes the fmax score using Cafa Assesment"
    _order = -20
    def __init__(self, learn):
        super().__init__(learn)
    
    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(
            [
                'bpo-NK-p', 'bpo-NK-f', 'bpo-LK-f', 'bpo-LK-f',
                'cco-NK-p', 'cco-NK-f', 'cco-LK-f', 'cco-LK-f',
                'mfo-NK-p', 'mfo-NK-f', 'mfo-LK-f', 'mfo-LK-f',
            ])

def addCafaAssesment(learn_cls):
    # TODO: First, I need to save the validation set in a groundtruth folder (for this I need to 
    # use the train_df and save just leaf_go of each category in a seperate file)
    # Second, I should write an script to save my validation result in a file (I should store the validations in each batch_end)
    # and then creating a .yaml config file and running the cafa assesment main file to produce results. If I Like to have them in tensorboard I should 
    # write a parser to read the result file and add them as a metric to my learner.
    
    # learn_cls.callbacks.append(CafaAssesment(learn_cls))
    pass
