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
- Clone cafa_assesment and make cafa_assesment importable
- Save Training and Validation as benchmark file
- Create a Callback for cafa_assesment  ... (will be run after training and validation,I hope to get perfect score after training!)

- Add taxonomy to the cafa sequence (with ancestors)
- Different Language Modeling

- (I wish I run the language modeling for the proteinet sooner on the Pine) Q: tax ans? Hummm sentencePiece? No

- positiveweight
- loss without flattening
- weight for loss

openconnect --user=panahia --script-tun --script "~/ocproxy/ocproxy -L 2222:pine.cs.vcu.edu:22 -D 11080" ramsvpn.vcu.edu

rsync -zavu -e "ssh -p2222 -i ~/.ssh/id_rsa3"  panahia@127.0.0.1:/home/panahia/projects/cafa/tensorboard-logs/ /home/panahia/projects/cafa/tensorboard-logs/

rsync -zavu -e "ssh -p2222 -i ~/.ssh/id_rsa3" /home/panahia/projects/cafa/data/cafa3/  panahia@127.0.0.1:/home/panahia/projects/cafa/data/cafa3/


mkdir /home/panahia/projects/cafa/log/
mkdir ./data/sprot_lm/data_cls/



