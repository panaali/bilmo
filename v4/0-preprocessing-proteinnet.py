#!/usr/bin/env python
# coding: utf-8
"""
Convert proteinnet fasta file to .csv file
"""

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from Bio import SeqIO
from ttictoc import TicToc


# In[2]:


t = TicToc() ## TicToc("name")
t.tic();

fasta_file = 'proteinnet12'
fasta_path = '../data/proteinnet/'

col_ids = []
col_seqs = []
for seq_record in SeqIO.parse(fasta_path + fasta_file, "fasta"):
    col_ids.append(seq_record.id)
    col_seqs.append(str(seq_record.seq))
cols = {'uni': col_ids, 'sequence': col_seqs}

t.toc();
print(t.elapsed)


# In[3]:


t = TicToc() ## TicToc("name")
t.tic();
df = pd.DataFrame(data=cols)
t.toc();
print(t.elapsed)


# In[4]:


df.to_csv('../data/proteinnet/proteinnet12.csv', index=False)


# In[5]:


len(df)


# In[ ]:




