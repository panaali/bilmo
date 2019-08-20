from src.scripts.config import Config
import numpy as np
import torch
import random
import logging
import os
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
    if conf['vocab_path'] and not os.path.exists(conf['vocab_path']):
        os.makedirs(conf['vocab_path'])
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

def initialize():
    config_logger(logPath=conf['log_path'], filename=conf['log_filename'])
    log.debug('Initializing')
    log.info(pformat(conf))  # Print config
    set_random_seed(conf['random_seed'])
    check_folder_path()


