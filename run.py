from src.scripts.config import Config
from src.scripts.initialize import initialize
from src.dataset.prepare_training_dataset import load_data
from src.dataset.create_databunch import create_databunch, get_cached_data_cls, print_data_cls_info
from src.learner.create_learner import create_learner
import logging

log = logging.getLogger("cafa-logger")
conf = Config.conf

if __name__ == "__main__":
    initialize()
    data_cls = get_cached_data_cls()
    if data_cls is None:
        df_train, df_valid = load_data()
        data_cls = create_databunch(df_train, df_valid)
    print_data_cls_info(data_cls)
    learn_cls = create_learner(data_cls)

    lr = 2e-2
    log.info('freeze')
    learn_cls.freeze()
    learn_cls.fit_one_cycle(1, lr, moms=(0.8, 0.7))
    # if not skip:
    learn_cls.fit_one_cycle(4, lr, moms=(0.8, 0.7))
    #
    learn_cls.save('cls-v1-0-' + conf['datetime'])

    log.info('unfreeze')
    learn_cls.freeze_to(-2)
    learn_cls.fit_one_cycle(2, slice(lr/(2.6**4), lr), moms=(0.8, 0.7))
    learn_cls.save('cls-v1-1-' + conf['datetime'])

    learn_cls.freeze_to(-3)
    learn_cls.fit_one_cycle(2, slice(lr/2/(2.6**4), lr/2), moms=(0.8, 0.7))
    learn_cls.save('cls-v1-2-' + conf['datetime'])

    learn_cls.unfreeze()
    learn_cls.fit_one_cycle(4, slice(lr/10/(2.6**4), lr/10), moms=(0.8, 0.7))
    learn_cls.save('cls-v1-3-' + conf['datetime'])

    learn_cls.fit_one_cycle(20, slice(lr/10/(2.6**4), lr/10), moms=(0.8, 0.7))
    learn_cls.save('cls-v1-4-' + conf['datetime'])
