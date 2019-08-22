# use run-classifier.py and run-language-modeler.py for running the modeling and classifier for now
# remove data_cls.pickle from the sprot_lm/data_cls folder if data block gives you error.
CWD="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if pgrep -x "gedit" > /dev/null
then
    echo "Tensorboard is running"
else
    echo "Starting Tensorboard"
    nohup tensorboard --port 6006 --logdir $CWD/tensorboard-logs >> $CWD/tensorboard-logs/tensorboard-app.log 2>&1 < /dev/null &
fi

if [ -d $CWD/CAFA_assessment_tool ] 
then
    echo "CAFA_assessment_tool is cloned" 
else
    echo "Cloning CAFA_assessment_tool"
    git clone https://github.com/ashleyzhou972/CAFA_assessment_tool.git
fi

latest_result="$(ls -tp ./data/sprot_lm/result/ | grep -v /$ | head -1)"
cp -rf $CWD/data/sprot_lm/result/$latest_result ./CAFA_assessment_tool/ZZZ_1_9606.txt
cd ./CAFA_assessment_tool

# Result:
python assess_main.py config.yaml

# Plotting
eval "$(conda shell.bash hook)"

conda deactivate
conda activate bilmo-py2
python plot.py config.yaml
conda deactivate
conda activate bilmo
cd ..

# FASTAI_HOME=$CWD
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# python -m fastai.launch \
# --gpus=0  $CWD/bilmo/classifier.py "./data/cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p" \
# --max_cpu_per_dataloader=1 \
# --bs=2 \
# --val_bs=4 \
# --fp16=0 \
# --use_sp_processor=0  \
# --sp_model=./data/sprot_lm/tmp/spm.model \
# --sp_vocab=./data/sprot_lm/tmp/spm.vocab \
# --sequence_col_name sequence \
# --label_col_name selected_go \
# --benchmarking 0 \
# --selected_go GO:0036094 \
# --tokenizer_n_char 1 \
# --valid_split_percentage 0.1 \
# --network Transformer
# --skip 0
# --lm_encoder lm-sp-ans-v1-5-enc  \
# --vocab_path ./data/sprot_lm/vocab_lm_sproat_seq_anc_tax.pickle \


# python -m torch.distributed.launch \
# --nproc_per_node=1 \
# $CWD/bilmo/classifier.py "./data/cafa3/CAFA 3 Protein Targets/CAFA3_training_data/cafa_train_enhanced.p" \
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
