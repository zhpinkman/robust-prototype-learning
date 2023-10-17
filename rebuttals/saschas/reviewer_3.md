**We thank the reviewer for the constructive feedback and the good assessment of our submitted work. We will incorporate the feedback in the revised version of the paper. We now clarify the questions raised by the reviewer:**


### Reasons To Reject Point 1:

The purpose of this paper is to consolidate the existing design choices and theories from the literature on prototype-based networks and create a framework for the evaluation of the robustness of these family of models on text classification tasks. Hence, to have extensive coverage, we include common practices from both prototype-based networks and robustness literature. This practice was also appreciated by the other reviewers in their reasons for acceptance.


### Reasons To Reject Point 2:

We acknowledge the concern of the reviewer about more comprehensive perturbation strategies and thank them for their valuable feedback; however, for the purpose of this study and only to achieve generalizable perturbations that can solely show the effect of perturbations on language models, we use three different models as target models when gathering the perturbations to ensure generalizability of our perturbations, and this is based on common practice (done by the authors of ADVGlue as well). 


### Reasons To Reject Point 3:

The main point of this paper is to utilize prototype-based networks, check if their inherent robustness properties will be transferred to text classification tasks as well, and make comparisons with language models that do not utilize prototypes. Achieving robustness by adversarial training requires access to the perturbations and extra training resources, which is not the case in our study, and prototype-based networks are more favorable in this regard. We will add a discussion about this point in the paper.

### Reasons To Reject Point 4:

We thank the reviewer for pointing out this important aspect of our study. The classification in PBNs is done based on the distance of examples to prototypes, and we claim that they have two advantages: 1. Since the model is only allowed to leverage the distances to prototypes, perturbations have a less direct effect on the outcome of the prediction, which otherwise, would’ve been a more direct effect on the embedding space; 2. Distances, as suggested by the vision community [1], can help identify perturbations even if recovering the correct prediction is difficult, which helps the model/user understand if the model is facing a perturbed example. We plan to add a comprehensive discussion of interpretability to the study both in terms of what prototypes represent (our unpublished experiments show that prototypes capture prominent semantic features of the dataset, like dates and location entities) and how they change by perturbations (utilizing PBNs, the change in the prediction outcome is reflected in the change in prototypes and model keeps its interpretability properties even under effective perturbations).

[1] Soares, E., Angelov, P., & Suri, N. (2022). Similarity-based Deep Neural Network to Detect Imperceptible Adversarial Attacks. 2022 IEEE Symposium Series on Computational Intelligence (SSCI), 1028–1035. https://doi.org/10.1109/SSCI51031.2022.10022016


### Questions For The Authors Point 1:

We thank the reviewer for their thorough review and for mentioning this point. Upon requesting the validation set of AdvGLUE SST-2 from the authors that would contain the type of perturbations, 17 examples were found to be human-generated and contained mixed perturbation types (not necessarily only character, word, or sentence-based). Furthermore, The human annotations do not have original sentences because the sentences were created specifically as an attack and are not manipulations of the original text; hence, we discarded those data points, proceeding with the left 131 examples. 


### Questions For The Authors Point 2:

We used BERT, RoBERTa, and DistilBERT to generate the perturbations that are generalizable enough (perturbations can change three different models' predictions) that could represent perturbations' effect on a variety of models. We did not use BERT-base as a backbone for prototype-based reasoning models since we used BERT-base models as the target model for our perturbations. This separation ensures that our evaluation framework is generalizable and doesn't put PBNs nor their backbones in a directly targeted spot by perturbations. Nevertheless, for the sake of the completeness of our experiments, we conducted additional experiments using BERT as a backbone, and their performance aligned with the results using other backbones reported in the paper. We will include these additional results in the paper. 


|           |     IMDB |          |          |          |  AG News |          |          |          |  DBPedia |          |          |          |    SST-2 |          |
|----------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| **Model** | **Orig** | **Char** | **Word** | **Sent** | **Orig** | **Char** | **Word** | **Sent** | **Orig** | **Char** | **Word** | **Sent** | **Orig** |  **Adv** |
|    BERT-S |     96.4 |     84.6 | **86.4** | **91.3** |     94.4 |     73.4 | **79.9** | **88.1** |     97.9 |     58.8 |     57.9 | **97.8** |     84.7 |     40.7 |
|    \+ PBN |     91.0 | **84.7** |     86.0 |     87.3 |     91.9 | **74.5** |     79.0 |     84.9 |     98.9 | **67.4** | **73.5** |     97.0 |     76.2 | **44.0** |
|       *Δ* |     -5.4 |     +0.1 |     -0.4 |     -4.0 |     -2.5 |     +1.1 |     -0.9 |     -3.2 |     +1.0 |     +8.6 |    +15.6 |     -0.8 |     -8.5 |     +3.3 |
|    BERT-M |     94.7 | **84.2** |     85.5 | **92.2** |     93.9 |     71.1 |     78.8 |     88.3 |     98.4 |     66.2 |     60.5 | **98.0** |     83.9 |     40.9 |
|    \+ PBN |     90.5 |     83.5 | **85.6** |     85.9 |     92.1 | **78.4** | **80.2** | **88.5** |     96.5 | **69.0** | **75.5** |     97.4 |     77.8 | **46.3** |
|       *Δ* |     -4.2 |     -0.7 |     +0.1 |     -6.3 |     -1.8 |     +7.3 |     +1.4 |     +0.2 |     -1.9 |     +2.8 |    +15.0 |     -0.6 |     -6.1 |     +5.4 |

Please note that the raised question is similar to Reasons To Reject Point 2 of Reviewer 2 and partially similar to Reasons To Reject Point 3 of Reviewer 1.

### Typos Grammar Style And Presentation Improvements Point 1:

We thank the reviewer for pointing us to this issue. References to LLMs will be corrected where necessary, and the distinction between pre-trained language models and large language models will be made in the paper.


### Typos Grammar Style And Presentation Improvements Point 2:

We do not use the WordSwapEmbedding in the textBugger implementation; however, we will check all the perturbations again to remove any potential doubts.
