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
```
Create the tokenizer using bpemb as belllow:

```import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import TabularDataset, Field, Iterator, BucketIterator, ReversibleField

import pyarabic.araby as araby
import pyarabic.number as number
```

Then create fields, the model expects data to be fed with in fromat of the batch dimension first, so we use batch_first = True:

```import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import TabularDataset, Field, Iterator, BucketIterator, ReversibleField

import pyarabic.araby as araby
import pyarabic.number as number
```

Next, load the dataset and build the vocabulary:

```SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
```
