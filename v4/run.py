print(2)
"""
What I need? a Makefile for data prepration.
Do things that don't scale
You may create things that never would be used
start with bare minimum that just solve the problem

- Downloading the data - I can solve this by downloading the files manually and 
    putting them into the right folder.
- Create preprocessing file. Prepare the csv files.
- LM with config.json
- CLS with config.json





reading options or loading option file

Downloading Datasets
Preprocessing the Datasets
LM training
Classfier training
Predicting
    Saving Prediction
    Converting Predicition to right format
    Checking Prediction format
Assesment Evaluation

Preprocessing Options 
    LM [just seq with tax_ids]
         from Uniprac(-) or Sprot
    Classifer
    Which sprot
    Cafa3 train or Extract from sprot
       Evidences for extraction

Check data folder or Download Sprot
Sprot to Dic
Dic to Pandas

Load Cafa3 train
Create my train from sprot using evidence list

Combine Cafa3 train with sprot for tax

load tax and go obo
expand Cafa3 train with tax_ancesstors and go_ancesstors

Create my train from sprot using evidence list



LM Training:
    


Classifier options
which training_set
to add (tax, tax_anc, none) to seq
to add go_anc, just leaf
Classifier Hyperparameters and Networks

Classifier Training:
   combine tax_ancesstors with seq for X
   combine go_ancesstors for Y 
"""
