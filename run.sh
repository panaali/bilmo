CWD="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FASTAI_HOME=$CWD

python -m fastai.launch \
--gpus=01 $CWD/src/classifier.py "./data/cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p" \
--max_cpu_per_dataloader=40 \
--bs=32 \
--fp16=0 \
--use_sp_processor=1  \
--sp_model=./data/sprot_lm/tmp/spm.model \
--sp_vocab=./data/sprot_lm/tmp/spm.vocab \
--lm_encoder lm-sp-ans-v1-3-enc2019-08-01_18-17-18  \
--sequence_col_name seq_anc_tax \
--label_col_name selected_go \
--selected_go GO:0017076 \
--benchmarking 0 \
