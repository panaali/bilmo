#!/usr/bin/env python
# coding: utf-8
"""
Convert CAFA3 target files to dataframe
"""
# %%
import pandas as pd
from glob import glob
from Bio import SeqIO

# %%
data_path = './data/'
target_path = data_path + 'cafa3/CAFA 3 Protein Targets/CAFA3_targets/Target files/'
for file in glob(target_path + "*.fasta"):
    col_ids = []
    prot_names = []
    col_seqs = []
    for seq_record in SeqIO.parse(file, "fasta"):
        col_ids.append(seq_record.id)
        desc = seq_record.description.split(' ')
        protein_name = desc[1] if len(desc) > 1 else None
        prot_names.append(protein_name)
        col_seqs.append(str(seq_record.seq))
        cols={'uniq_id': col_ids, 'sequence': col_seqs, 'prot_names': prot_names}

# %%
df = pd.DataFrame(data=cols)
print('len(df)', len(df))
# %%
df.to_csv(data_path + 'cafa3/targets.p', index=False)





#%%
