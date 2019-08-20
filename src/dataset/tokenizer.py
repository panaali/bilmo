from src.scripts.config import Config
from fastai.text.transform import BaseTokenizer
from textwrap import wrap
conf = Config.conf

class dna_tokenizer_n_char(BaseTokenizer):

    def tokenizer(self, t):
        tokens = t.split(' ')
        begin_of_sentence = tokens[0]
        seq = tokens[1]
        after_seq = tokens[2:-1]    # taxon ancestors
        end_of_sentence = tokens[-1]
        result = [begin_of_sentence]
        # sequence string to list
        result += wrap(seq, conf['tokenizer_number_of_char'])
        result += after_seq
        result.append(end_of_sentence)
        return result
