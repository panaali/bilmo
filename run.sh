CWD="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FASTAI_HOME=$CWD
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
python -m fastai.launch \
--gpus=0  $CWD/src/classifier.py "./data/cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p" \
--max_cpu_per_dataloader=1 \
--bs=32 \
--fp16=0 \
--use_sp_processor=0  \
--sp_model=./data/sprot_lm/tmp/spm.model \
--sp_vocab=./data/sprot_lm/tmp/spm.vocab \
--sequence_col_name sequence \
--label_col_name selected_go \
--benchmarking 0 \
--selected_go GO:0036094 \
--tokenizer_n_char 2 \
# --lm_encoder lm-sp-ans-v1-5-enc  \
# --vocab ./data/sprot_lm/vocab_lm_sproat_seq_anc_tax.pickle \


# python -m torch.distributed.launch \
# --nproc_per_node=1 \
# $CWD/src/classifier.py "./data/cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p" \
# --max_cpu_per_dataloader=1 \
# --bs=32 \
# --fp16=0 \
# --use_sp_processor=0  \
# --sp_model=./data/sprot_lm/tmp/spm.model \
# --sp_vocab=./data/sprot_lm/tmp/spm.vocab \
# --lm_encoder lm-sp-ans-v1-5-enc  \
# --vocab ./data/sprot_lm/vocab_lm_sproat_seq_anc_tax.pickle \
# --sequence_col_name sequence \
# --label_col_name selected_go \
# --selected_go GO:0036094 \
# --benchmarking 0 \
