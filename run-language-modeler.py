from bilmo.scripts.config import Config
from bilmo.scripts.initialize import initialize
from bilmo.dataset.prepare_training_dataset import load_data
from bilmo.dataset.create_databunch import create_databunch, get_cached_data, print_data_cls_info
from bilmo.learner.create_learner import create_learner
from bilmo.learner.test_cafa import test_cafa
from bilmo.callbacks.tensorboard import add_tensorboard
import logging

log = logging.getLogger("cafa-logger")
conf = Config.conf

if __name__ == "__main__":
    initialize()
    data_cls, data_test = get_cached_data()
    if data_cls is None:
        df_train, df_valid = load_data()
        data_cls, data_test = create_databunch(df_train, df_valid)
    print_data_cls_info(data_cls)
    learn_cls = create_learner(data_cls)

    if conf['add_tensorboard']:
        add_tensorboard(learn_cls)

    if not conf['skip_training']:
        lr = 2e-2
        log.info('freeze')
        learn_cls.freeze()
        learn_cls.fit_one_cycle(5, lr, moms=(0.8, 0.7))
        # learn_cls.save('cls-v1-0-' + conf['datetime'])

        log.info('unfreeze')
        learn_cls.freeze_to(-2)
        learn_cls.fit_one_cycle(2, slice(lr/(2.6**4), lr), moms=(0.8, 0.7))
        # learn_cls.save('cls-v1-1-' + conf['datetime'])

        learn_cls.freeze_to(-3)
        learn_cls.fit_one_cycle(2, slice(lr/2/(2.6**4), lr/2), moms=(0.8, 0.7))
        # learn_cls.save('cls-v1-2-' + conf['datetime'])

        learn_cls.unfreeze()
        learn_cls.fit_one_cycle(
            4, slice(lr/10/(2.6**4), lr/10), moms=(0.8, 0.7))
        # learn_cls.save('cls-v1-3-' + conf['datetime'])

        learn_cls.fit_one_cycle(
            20, slice(lr/10/(2.6**4), lr/10), moms=(0.8, 0.7))
        learn_cls.save('cls-v1-4-' + conf['datetime'])
        if conf['export_model']:
            learn_cls.export(file='export/' +
                             'export-cls-v1-4-' + conf['datetime'] + '.pkl')

        log.info('Done Training')
    else:
        log.info('Skipped training')
    if conf['full_validation']:
        # we set our full validation as test_dl
        valid_result = learn_cls.validate(learn_cls.data.test_dl)
        log.info('full validation: loss, acc, f_beta' + str(valid_result))

    if conf['test_on_cafa3_testset']:
        log.info('Start Test Prediction')
        test_cafa(data_test, learn_cls)
