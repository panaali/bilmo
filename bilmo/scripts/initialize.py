from bilmo.scripts.config import Config
import numpy as np
import torch
import random
import logging
import os
import subprocess
from pprint import pformat
conf = Config.conf
log = logging.getLogger("cafa-logger")

def set_random_seed(seed):
    if isinstance(seed, int):
        log.info('Setting random seed to ' + str(seed))
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)
        random.seed(seed)

def check_folder_path():
    local_project_path = conf['local_project_path']
    if not os.path.exists(local_project_path):
        os.makedirs(local_project_path)
    log.debug('local_project_path:' + local_project_path)

def config_logger(logPath, filename):
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, filename))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    log.setLevel(conf['log_level'])


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def initialize():
    config_logger(logPath=conf['log_path'], filename=conf['log_filename'])
    log.debug('Initializing')
    log.debug('Git Hash: ' + str(get_git_revision_hash()) + ' - short hash: ' +
              str(get_git_revision_short_hash()))
    log.info(pformat(conf))  # Print config
    set_random_seed(conf['random_seed'])
    check_folder_path()
    import sys
    is_debugger = True if getattr(sys, 'gettrace', None) is not None else False
    if is_debugger:
        log.debug('Debug mode detected, setting n_workers to 0')
        conf['n_workers'] = 0
