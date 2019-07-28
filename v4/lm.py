from fastai.script import *
from fastai.metrics import *
from fastai.text import *
from fastai import *
from fastai.distributed import *

import os
try:
	os.chdir(os.path.join(os.getcwd(), 'v3'))
	print(os.getcwd())
except:
	pass


class dna_tokenizer(BaseTokenizer):
    def tokenizer(self, t):
    #         return list(t)
        res = []
        tokens = t.split(' ')
        before_seq = tokens[:-2]
        seq = tokens[-2]
        eos = tokens[-1]

        res = before_seq
        res += list(seq)  # sequence string to list
        res.append(eos)

        return res


@call_parse
def main(gpu: Param("GPU to run on", str) = None):
    #### Distributed
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
    workers = min(16, num_cpus()//n_gpus)
    print(gpu, 'n_gpus', n_gpus)
    print(gpu, 'workers', workers)

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int)
    # args = parser.parse_args()
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    ####



    local_path = '../'

    """## Prepare fastai"""
    # torch.cuda.set_device(2)
    np.random.seed(0)
    """## Prepare Dataset"""
    local_project_path = local_path + 'data/proteinnet/'
    if not os.path.exists(local_project_path):
        os.makedirs(local_project_path)
    print('local_project_path:', local_project_path)

    """## Tokenization"""


    tokenizer = Tokenizer(tok_func=dna_tokenizer, pre_rules=[],
                        post_rules=[], special_cases=[])
    processor = [TokenizeProcessor(tokenizer=tokenizer, include_bos=True,
                                include_eos=True), NumericalizeProcessor(max_vocab=30000)]
    # df_whole_sprot = pickle.load(
    #     open('../data/uniprot_sprot/sproat_parser_df_sp.p', 'rb'))

    # df_whole_sprot_seq = df_whole_sprot[['sequence']].copy()
    # df = df_whole_sprot_seq
    df_test = pd.read_csv('../data/proteinnet/test.csv')
    df_test.rename(columns={"seq": "sequence"}, inplace=True)
    df = df_test[:1000].copy()


    bs = 512

    data_lm = (TextList.from_df(df, path=local_project_path, cols='sequence', processor=processor)
                    .split_by_rand_pct(0.1, seed = 42)
                    .label_for_lm()
                    .databunch(bs=bs, num_workers=workers))

    # data_lm.vocab.save(local_project_path + 'lm-whole-sp-v2-vocab.pkl')

    print('data_cls Training set size', len(data_lm.train_ds))
    print('data_cls Validation set size', len(data_lm.valid_ds))
    print('vocab size ', len(data_lm.vocab.itos))

    learn_lm = language_model_learner(
        data_lm, AWD_LSTM, drop_mult=1, pretrained=False)#.to_distributed(args.local_rank)#.to_fp16()

    if gpu is None:
        print(gpu, 'DataParallel')
        learn_lm.model = nn.DataParallel(learn_lm.model)
    else:
        print(gpu, 'to_distributed')
        learn_lm.to_distributed(gpu)
    learn_lm.to_fp16()

    lr = 3e-3
    print(gpu, 'freeze')
    learn_lm.freeze()
    learn_lm.fit_one_cycle(1, lr, moms=(0.8, 0.7))
    print(gpu, 'unfreeze')
    learn_lm.unfreeze()
    learn_lm.fit_one_cycle(1, lr, moms=(0.8, 0.7))


    # learn_lm.load('lm-gpu3-sp-40M-v2');

    # learn_lm.freeze()
    # lr = 3e-3
    # learn_lm.unfreeze()
    # learn_lm.fit_one_cycle(1, lr, moms=(0.8, 0.7))
    # learn_lm.save('lm-whole-sp-dist-v1')
    # learn_lm.save_encoder('lm-whole-sp-dist-v1-enc')

    # learn_lm.fit_one_cycle(10, lr/10, moms=(0.8, 0.7))
    # learn_lm.save('lm-whole-sp-dist-v2')
    # learn_lm.save_encoder('lm-whole-sp-dist-v2-enc')

    # learn_lm.fit_one_cycle(10, slice(lr/2.6**4, lr/100), moms=(0.8, 0.7))
    # learn_lm.save('lm-whole-sp-dist-v3')
    # learn_lm.save_encoder('lm-whole-sp-dist-v2-enc')
    print('Done')


# main(None)
