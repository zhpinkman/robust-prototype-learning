**We thank the reviewer for the thoughtful questions, and for the constructive feedback. Below, we provide answers to the raised questions. For the camera-ready version, we plan to address these points directly in the discussion of the results to make our contribution stronger.**


### Reasons To Reject Point 1:

Please see the section Questions For The Authors, where we provide a thorough explanation in response to the reviewer's point. Additionally, addressing the concerns of the reviewer about the disparity between the robustness properties of PBNs using different backbones, we conducted additional experiments using BERT as the backbone as well, as mentioned in the next section. Comparing the results gathered from models with different sizes and also different embedding properties (Transformer models being superior to CNN models), we attribute the disparity to the embedding space properties of the backbone and consider a strong backbone more favorable to the robustness of PBNs.

### Reasons To Reject Point 2:

We used BERT, RoBERTa, and DistilBERT to generate the perturbations that are generalizable enough (perturbations can change three different models' predictions) that could represent perturbations' effect on a variety of models. We did not use BERT-base as a backbone for PBNs since we used BERT-base models as the target model for our perturbations. This separation ensures that our evaluation framework is generalizable and doesn't put PBNs nor their backbones in a directly targeted spot by perturbations. For the sake of the completeness of our experiments and according to the reviewer's suggestion, we conducted additional experiments using BERT as a backbone, whose findings aligned with those using other backbones reported in the paper. We will include these additional results in the paper.

|           |     IMDB |          |          |          |  AG News |          |          |          |  DBPedia |          |          |          |    SST-2 |          |
|----------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| **Model** | **Orig** | **Char** | **Word** | **Sent** | **Orig** | **Char** | **Word** | **Sent** | **Orig** | **Char** | **Word** | **Sent** | **Orig** |  **Adv** |
|    BERT-S |     96.4 |     84.6 | **86.4** | **91.3** |     94.4 |     73.4 | **79.9** | **88.1** |     97.9 |     58.8 |     57.9 | **97.8** |     84.7 |     40.7 |
|    \+ PBN |     91.0 | **84.7** |     86.0 |     87.3 |     91.9 | **74.5** |     79.0 |     84.9 |     98.9 | **67.4** | **73.5** |     97.0 |     76.2 | **44.0** |
|       *Δ* |     -5.4 |     +0.1 |     -0.4 |     -4.0 |     -2.5 |     +1.1 |     -0.9 |     -3.2 |     +1.0 |     +8.6 |    +15.6 |     -0.8 |     -8.5 |     +3.3 |
|    BERT-M |     94.7 | **84.2** |     85.5 | **92.2** |     93.9 |     71.1 |     78.8 |     88.3 |     98.4 |     66.2 |     60.5 | **98.0** |     83.9 |     40.9 |
|    \+ PBN |     90.5 |     83.5 | **85.6** |     85.9 |     92.1 | **78.4** | **80.2** | **88.5** |     96.5 | **69.0** | **75.5** |     97.4 |     77.8 | **46.3** |
|       *Δ* |     -4.2 |     -0.7 |     +0.1 |     -6.3 |     -1.8 |     +7.3 |     +1.4 |     +0.2 |     -1.9 |     +2.8 |    +15.0 |     -0.6 |     -6.1 |     +5.4 |

Please note that the raised question is similar to Questions For The Authors Point 2 of Reviewer 3 and partially similar to Reasons To Reject Point 3 of Reviewer 1.

### Questions For The Authors Point 1:

The clustering loss is an additional regularization term in the overall loss function. As usual with regularization terms, the term forces the network to learn a trade-off between the competing goals. In our case, all loss terms, including the accuracy target loss, are somewhat competing goals. Therefore, the absence of a regularization term could cause an improvement of the other loss terms. The clustering loss forces the backbone to project all samples to be close to a prototype in the embedding space, an objective that is useful for interpretability because it ensures that the closest prototype serves as a valid proxy for the sample. However, this loss term lowers the possible achievable accuracy because forcing each sample to be close to a prototype lowers the diversity in the embedding space, as our observations suggest as well: If we select a prototype and compute the distances between the prototype and each sample in the embedding space, the mean and standard deviation over the distances can be used to describe the spread of the embedded data points around the prototype. These values are -1.8120(e-07)±1.4924(e-06) with the clustering loss and -2.4093(e-08)±1.7166(e-07) without, confirming the hypothesis from above (the mean distance without the clustering loss is larger).

Please note that the raised question is partially similar to Reasons To Reject Point 4 of Reviewer 1.


### Questions For The Authors Point 2:

During training, the optimal number of prototypes was automatically selected. From the literature [1], it is known that the optimal number of prototypes is non-trivial and not necessarily the number of classes or training data points. Therefore, the number 16 is in accordance with results already published and we found it empirically to perform best. The reason for the deterioration of performance with the increasing number of prototypes is that the model is forced to learn a more complex embedding space, which is more difficult. This is in accordance with the observation that the clustering loss lowers the achievable accuracy.

[1] Crammer, K., Gilad-Bachrach, R., Navot, A., & Tishby, N. (2002). Margin analysis of the LVQ algorithm. Advances in neural information processing systems, 15.


### Questions For The Authors Point 3:

In prototype-based learning, an appropriate selection of the dissimilarity measure is one of the major investigation subjects (note the different measures used in [1, 2]). Thus, the dissimilarity selection is usually treated as a hyperparameter. In our case, the Euclidean distance works better than the cosine similarity. One reason for the different performance might be that the Euclidean distance forces the embedded samples to be close to a point (the prototype) in the vector space, whereas the cosine similarity forces embedded samples to be in the same direction as the prototype. Consequently, the formation of clusters is completely different from the two measures, leading to the observed performance differences.

[1] Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019). This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems, 32.

[2] Mettes, P., Van der Pol, E., & Snoek, C. (2019). Hyperspherical prototype networks. Advances in neural information processing systems, 32.