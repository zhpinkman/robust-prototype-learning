**We thank the reviewer for the feedback and insightful suggestions. We are pleased that the paper idea is overall positively perceived by the reviewer. We now clarify the questions raised by the reviewer:**


### Reasons To Reject Point 1:

Thank you for spotting this. After receiving this feedback, we analyzed our figure and explanation, and we agree that we have to improve it. We will improve it in the camera-ready version of the paper and attach our revised explanation here for your reference: 

>_"PBNs are a family of interpretable models that classifies data points based on their similarity to certain prototypes learned during training. The prototypes summarize the prominent semantic patterns of the dataset, and being close or far from these semantic patterns adjusts the prediction outcome in PBNs. This property is achieved through two mechanisms: 1. Prototypes are defined in the same embedding space as input examples, which makes them interpretable by leveraging input examples close to them; 2. Prototypes are designed to cluster semantically similar training examples, which makes them representative of the prominent patterns embedded in the dataset and input examples. The decisions made by PBNs are inherently interpretable because prototypes are trained to be aligned with previous observations. This property helps users to understand the behavior of the model during inference better by looking at the closest activated prototypes."_


### Reasons To Reject Point 2:

Again, we thank the reviewer for spotting this and are sorry for the inconvenience. Models used as targets to gather generalizable perturbations were BERT-base, RoBERTa-base, and DistilBERT-base, fine-tuned on three datasets separately for getting the perturbations for each dataset. BERT models used as baselines were BERT-medium [1] and BERT-small [2], which are the smaller pre-trained BERT variants chosen with respect to having a closer number of parameters to PBN variants analyzed in our study. 

We only used the encoder part of BART architecture for our experiments, and that’s the reason behind it being smaller than usual BART models. The BART and ELECTRA models used in our main experiments were BART-base and ELECTRA-base, and the bigger variant of BART that we used to test the effect of scaling up the backbones is the BART-large encoder. In terms of pre-training, the models used as target models to gather generalizable perturbations were fine-tuned on three datasets separately prior to getting the perturbations for each dataset. PBN models and their backbones were all fine-tuned on each dataset separately (backbones were not pre-trained further than their original pre-training and were fine-tuned in the end-to-end fine-tuning of the PBN model as a whole).

We will include these and related details of our experimental setup in the final version of the paper. 

[1] https://huggingface.co/prajjwal1/bert-medium

[2] https://huggingface.co/prajjwal1/bert-small 


### Reasons To Reject Point 3:

The study of scaling up the transformer backbones was intended to test their effectiveness when used as a backbone for PBNs; hence, we used BART-base and BART-large as PBN’s backbones for this experiment. BERT-small and BERT-medium were only used as baselines to incorporate models that are from the BERT family but closer to PBN models in terms of the number of parameters for a more fair comparison. Nevertheless, we also conducted experiments of scaling up the backbones using BERT-base and BERT-medium, and their results align with the results reported in the paper. We will include these results in the final version of the paper.


|           |     IMDB |          |          |          |  AG News |          |          |          |  DBPedia |          |          |          |    SST-2 |          |
|----------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| **Model** | **Orig** | **Char** | **Word** | **Sent** | **Orig** | **Char** | **Word** | **Sent** | **Orig** | **Char** | **Word** | **Sent** | **Orig** |  **Adv** |
|    BERT-S |     96.4 |     84.6 | **86.4** | **91.3** |     94.4 |     73.4 | **79.9** | **88.1** |     97.9 |     58.8 |     57.9 | **97.8** |     84.7 |     40.7 |
|    \+ PBN |     91.0 | **84.7** |     86.0 |     87.3 |     91.9 | **74.5** |     79.0 |     84.9 |     98.9 | **67.4** | **73.5** |     97.0 |     76.2 | **44.0** |
|       *Δ* |     -5.4 |     +0.1 |     -0.4 |     -4.0 |     -2.5 |     +1.1 |     -0.9 |     -3.2 |     +1.0 |     +8.6 |    +15.6 |     -0.8 |     -8.5 |     +3.3 |
|    BERT-M |     94.7 | **84.2** |     85.5 | **92.2** |     93.9 |     71.1 |     78.8 |     88.3 |     98.4 |     66.2 |     60.5 | **98.0** |     83.9 |     40.9 |
|    \+ PBN |     90.5 |     83.5 | **85.6** |     85.9 |     92.1 | **78.4** | **80.2** | **88.5** |     96.5 | **69.0** | **75.5** |     97.4 |     77.8 | **46.3** |
|       *Δ* |     -4.2 |     -0.7 |     +0.1 |     -6.3 |     -1.8 |     +7.3 |     +1.4 |     +0.2 |     -1.9 |     +2.8 |    +15.0 |     -0.6 |     -6.1 |     +5.4 |

Please note that the raised question is partially similar to Reasons To Reject Point 2 of Reviewer 2 and Questions For The Authors Point 2 of Reviewer 3.


### Reasons To Reject Point 4:

The goal of this study is to provide a framework for evaluating the robustness of PBNs by incorporating common design choices in literature and not necessarily achieving the highest robustness. Furthermore, all the objective functions included in the framework are proven to be essential for the interpretability of PBNs that, if otherwise discarded, eliminate any reason for using PBNs. On this ground, we only consider eliminating objective functions and testing their effect on robustness in their own regard and their own ablation study. 

In terms of explanation as to why clustering loss hurts the robustness of PBNs, we refer the reviewer to the following statement: The clustering loss is an additional regularization term in the overall loss function. As usual with regularization terms, the term forces the network to learn a trade-off between the competing goals. In our case, all loss terms, including the accuracy target loss, are somewhat competing goals. Therefore, the absence of a regularization term could cause an improvement of the other loss terms. The clustering loss forces the backbone to project all samples to be close to a prototype in the embedding space, an objective that is useful for interpretability because it ensures that the closest prototype serves as a valid proxy for the sample. However, this loss term lowers the possible achievable accuracy because forcing each sample to be close to a prototype lowers the diversity in the embedding space, as our observations suggest as well: If we select a prototype and compute the distances between the prototype and each sample in the embedding space, the mean and standard deviation over the distances can be used to describe the spread of the embedded data points around the prototype. These values are -1.8120(e-07)±1.4924(e-06) with the clustering loss and -2.4093(e-08)±1.7166(e-07) without, confirming the hypothesis from above (the mean distance without the clustering loss is larger).

Please note that the raised question is partially similar to Questions For The Authors Point 1 of Reviewer 2.

### Questions For The Authors:

The raised question here is similar to Point 2 in Reasons To Reject. We kindly refer to the provided answer in that section.