from fastai.metrics import accuracy, FBeta

f1 = FBeta(average='macro')
f1.beta = 1
