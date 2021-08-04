# Arabic-GEC
Neural-based automatic Arabic Grammar Error Correction (AGEC) model based on sequence-to-sequence multi-heads attentions Transformer. Initially, we introduce a an unsupervised method to generate a large-scale synthetic dataset based on confusion function to increase the amount of training set. The standard seq2seq Transformer is equipped with capsule network to aggregate linguistic features cross layers dynamically we also added a regularization term in the training objective using Kullback-Leibler divergence to overcome to improve the agreement between R2L and L2R models.
# Model requirements
Regarding load and run the trained models it requires a working installation of following:
- Python 3.6.10 or latest 
- pytorch==1.6.0
- torchtext==0.6.0
- numpy==1.17.0
- pandas==1.17.0
- PyArabic==0.6.10
- bpemb==0.3.2
- nltk==3.3
