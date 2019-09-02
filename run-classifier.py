from bilmo.scripts.config import Config
conf = Config.conf
if conf['comet.ml']:
    from comet_ml import Experiment

import os
if conf['gpu']:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf['gpu'])
    import torch
    torch.cuda.set_device(0)

from bilmo.scripts.initialize import initialize
from bilmo.dataset.prepare_datasets_dataframe import load_data_train, load_data_test
from bilmo.dataset.create_databunch import create_databunch, get_cached_data, print_data_cls_info, print_data_test_info
from bilmo.learner.create_learner import create_learner
from bilmo.learner.test_cafa import test_cafa
from bilmo.callbacks.tensorboard import add_tensorboard
import logging

log = logging.getLogger("cafa-logger")


def train_without_pretraining(learn_cls):
    lr = 3e-3
    log.info('unfreeze')
    learn_cls.unfreeze()
    learn_cls.fit_one_cycle(1, lr, moms=(0.8, 0.7))

    if not conf['just_one_epoch']:
        learn_cls.fit_one_cycle(conf['num_epochs'], lr, moms=(0.8, 0.7))
        if conf['save_model']:
            learn_cls.save('cls-v1-0-' + conf['datetime'])

        if conf['export_model']:
            learn_cls.export(file='export/' + 'export-cls-v1-4-' +
                             conf['datetime'] + '.pkl')  # TODO: use hash of config
    else:
        log.info('just_one_epoch is True')


def train_with_pretraining(learn_cls):
    lr = 2e-2
    log.info('freeze')
    learn_cls.freeze()
    learn_cls.fit_one_cycle(10, lr, moms=(0.8, 0.7))

    if not conf['just_one_epoch']:
        learn_cls.fit_one_cycle(10, lr, moms=(0.8, 0.7))
        # learn_cls.save('cls-v1-0-' + conf['datetime'])

        log.info('unfreeze -2')
        learn_cls.freeze_to(-2)
        learn_cls.fit_one_cycle(10, slice(lr/(2.6**4), lr), moms=(0.8, 0.7))
        # learn_cls.save('cls-v1-1-' + conf['datetime'])

        log.info('unfreeze -3')
        learn_cls.freeze_to(-3)
        learn_cls.fit_one_cycle(conf['num_epochs'],
                                slice(lr / 2 / (2.6**4), lr / 2),
                                moms=(0.8, 0.7))
        # learn_cls.save('cls-v1-2-' + conf['datetime'])

        log.info('unfreeze')
        learn_cls.unfreeze()
        learn_cls.fit_one_cycle(10, slice(lr/10/(2.6**4), lr/10), moms=(0.8, 0.7))
        # learn_cls.save('cls-v1-3-' + conf['datetime'])

        learn_cls.fit_one_cycle(10,
                                slice(lr / 10 / (2.6**4), lr / 10),
                                moms=(0.8, 0.7))
        if conf['save_model']:
            learn_cls.save('cls-v1-4-' + conf['datetime'])
        if conf['export_model']: # doesn't work , should swap pickle with dill
            learn_cls.export(file = 'export/' + 'export-cls-v1-4-' + conf['datetime']+ '.pkl') # use hash of config
    else:
        log.info('just_one_epoch is True')



if __name__ == "__main__":
    initialize()
    data_cls, data_test = get_cached_data()
    df_test = load_data_test() # needed for cafa3_testing at the end
    if data_cls is None:
        log.info('No cached data found, loading data from df')
        df_train, df_valid = load_data_train()
        data_cls, data_test = create_databunch(df_train, df_valid, df_test)
    learn_cls = create_learner(data_cls)

    if conf['add_tensorboard']:
        add_tensorboard(learn_cls)

    if not conf['skip_training']:
        if conf['lm_encoder_path'] is None:
            log.info('Train without pretraining')
            train_without_pretraining(learn_cls)
        else:
            log.info('Train with pretraining')
            train_with_pretraining(learn_cls)

        log.info('Done Training')
    else:
        log.info('Skipped training')
    # print('full valid validation: loss, acc, f_beta', learn_cls.validate(learn_cls.data.test_dl))

    if conf['test_on_cafa3_testset']:
        log.info('Start Test Prediction')
        test_cafa(data_test, learn_cls, df_test)
    else:
        log.info('Skipping Test Prediction')

    log.info('The End - run-classifier.py')
