from src.scripts.config import Config
from src.dataset.tokenizer import dna_tokenizer_n_char
from fastai.text.data import TokenizeProcessor, NumericalizeProcessor, OpenFileProcessor, SPProcessor, TextDataBunch, TextClasDataBunch
from fastai.text.transform import BaseTokenizer, Vocab
from fastai.text import Tokenizer
import os.path

import logging
import dill as pickle

conf = Config.conf
log = logging.getLogger("cafa-logger")
def get_vocab():
    if conf['vocab_path'] is not None:
        vocab_obj = pickle.load(
            open(conf['local_project_path'] + conf['vocab_path'], 'rb'))
        vocab_class_obj = Vocab.load(
            conf['local_project_path'] + conf['vocab_path'])
        log.debug('vocab object loaded, len ' + str(len(vocab_obj)))
    else:
        vocab_obj = None
        vocab_class_obj = None
    return vocab_obj, vocab_class_obj


def get_processor(vocab_class_obj):
    if conf['use_sentencePiece']:
        processor = [OpenFileProcessor(), SPProcessor(
            sp_model=conf['sentencePiece']['model'], sp_vocab=conf['sentencePiece']['vocab'], max_sentence_len=conf['sentencePiece']['max_sentence_len'], max_vocab_sz=conf['sentencePiece']['max_vocab'])]
    else:
        tokenizer = Tokenizer(tok_func=dna_tokenizer_n_char, pre_rules=[],
                              post_rules=[], special_cases=[])
        processor = [TokenizeProcessor(tokenizer=tokenizer, include_bos=True,
                                       include_eos=True), NumericalizeProcessor(vocab=vocab_class_obj, max_vocab=conf['max_vocab'])]
    return processor, tokenizer


def get_cached_data_cls():
    if conf['use_cached_data_cls'] and os.path.exists(conf['data_cls_cached_location']):
            return pickle.load(open(conf['data_cls_cached_location'], 'rb'))
        
    return None


def save_data_cls(data_cls):
    if conf['data_cls_cached_location']:
        pickle.dump(data_cls , open(conf['data_cls_cached_location'],'wb'))

def save_vocab(data_cls):
    if conf['vocab_path'] is None:
        data_cls.vocab.save(conf['local_project_path'] + 'vocab/' +
                            'vocab_cls-' + conf['datetime'] + '.pickle')

def print_data_cls_info(data_cls):
    log.debug('sample x: ' + data_cls.train_ds.x[0].text)
    log.debug('sample y: ' + str(data_cls.train_ds.y[0]))
    log.debug('data_cls Training set size: ' + str(len(data_cls.train_ds)))
    log.debug('data_cls Validation set size: ' + str(len(data_cls.valid_ds)))
    log.debug('vocab size: ' + str(len(data_cls.vocab.itos)))

def create_databunch(train_df, valid_df):
    vocab_obj, vocab_class_obj = get_vocab()
    processor, tokenizer = get_processor(vocab_class_obj)

    if conf['smaller_valid_df'] is not None:
        valid_df_small = valid_df[:conf['smaller_valid_df']]
    else:
        valid_df_small = valid_df

    data_cls = TextClasDataBunch.from_df(conf['local_project_path'], train_df=train_df, valid_df=valid_df_small, test_df=valid_df,
                                         tokenizer=tokenizer, vocab=vocab_class_obj,
                                         text_cols=conf['sequence_col_name'],
                                         label_cols='selected_class', label_delim=None,
                                         max_vocab=conf['max_vocab'], min_freq=2, include_bos=True,
                                         include_eos=True,
                                         bs=conf['bs'], val_bs=conf['val_bs'], num_workers=conf['n_workers'])

    save_data_cls(data_cls)
    save_vocab(data_cls)
    return data_cls
