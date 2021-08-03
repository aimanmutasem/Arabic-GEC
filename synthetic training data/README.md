# Synthetic data
The constructed training and development sets are available in .csv format in the below links :

[AGEC_Training_set.csv - 5.91 GB](https://drive.google.com/file/d/1tZq05b453lDsSLDCHQztM9eCCgN_gEah/view?usp=sharing)

[AGEC_development_set.csv - 676 MB](https://drive.google.com/file/d/1_6YFDlSkR7ifJ0P2DLuysKzrHBzCYnit/view?usp=sharing)

# Preparing the Data

Import all the required modules and packages.
 
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import TabularDataset, Field, Iterator, BucketIterator, ReversibleField

import pyarabic.araby as araby
import pyarabic.number as number

import bpemb
from bpemb import BPEmb
bpemb_ar = BPEmb(lang="ar", vs=1000, dim=300) 
```
Create the tokenizer using bpemb as belllow:

```py
def normalizeString(line):
    line = araby.strip_tatweel(line)
    line = araby.strip_tashkeel(line)
    line = bpemb_ar.encode(line)
    return line
```

Then create fields, the model expects data to be fed with in fromat of the batch dimension first, so we use batch_first = True:

```py
SRC = Field(tokenize= normalizeString, init_token='<sos>', eos_token='<eos>',  batch_first=True) 
TRG = Field(tokenize= normalizeString, init_token='<sos>', eos_token='<eos>',  batch_first=True) 
```

Next, build the vocabulary and load the dataset:

````py
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

train_data, valid_data = TabularDataset.splits(path='../data/',train='AGEC_Training_set.csv',
    validation='AGEC_development_set.csv' , format='csv',
    fields=[('src', SRC), ('trg', TRG)], skip_header=True) 
