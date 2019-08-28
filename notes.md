# Todo:

- Add fmax instead of F1 at the end of training and validation (async prefered)
- Write validation as ground truth for assessment 
- Precision and Recall during training
- + Comet.ml with all config
- Language Modeler
- Export and import model and cafa test preds
- An script to prepare the servers (Aurora1, Aurora2, Tesla, Pine) All
- * Add Tom's Loss (Cross Entropy loss)
- Add Hparams


# Preparing the assesment tools:
git clone https://github.com/ashleyzhou972/CAFA_assessment_tool.git
mv ./CAFA_assessment_tool/precrec/benchmark/ ./CAFA_assessment_tool/precrec/benchmark-bak/
mkdir ./CAFA_assessment_tool/precrec/benchmark/
cp -r ./data/cafa3/CAFA\ 3\ Benchmarks/benchmark20171115/* ./CAFA_assessment_tool/precrec/benchmark/


sshAurora2 -L 6006:localhost:6006

Today:
- add theasholdlessf1 per epoch
- save Training and Validation as benchmark file
- Create a callback for cafa_assesment  ... (will be run after training and validation,I hope to get perfect score after training!)

