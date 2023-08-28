# REVIEW 3

>>We thank the reviewer for their feedback and insightful suggestions. We are pleased that the paper idea is overall positively perceived by the reviewer. We now clarify the questions raised by the reviewer.

## Paper Topic And Main Contributions:
This study examines the robustness and interpretability of prototype-based networks in text classification. To investigate the properties of PBNs regarding robustness, a framework is proposed, which includes backbone architectures, size, and objective functions. The evaluation on three benchmarks demonstrates the effectiveness of PBNs in enhancing robustness.

## Reasons To Accept:
The paper is well-written and easy-to-follow.

The proposed framework is comprehensive and covers the most important aspects.

The evaluation is extensive across different choices of components.

## Reasons To Reject:

1. Most components in the framework are existing methods, including components in PBN and perturbation generation methods in the evaluation protocol.

>> The purpose of this paper is to consolidate the existing design choices and theories from literature on prototype-based networks, and create a framework for evaluation of robustness of these family of models on text classification tasks, hence to have extensive coverage, we tried to include practices from both prototype-based network and robustness literature. We also point the reviewer to the first reason for acceptance from the other two reviewers, who share our point of view, and restate our motivation for this study.

---

2. As pointed out in [1], static evaluation cannot fully reflect the robustness of the model. This is also verified in the BERT-S and BERT-M results in Table 3, where a different BERT can correct classify the generated adversarial examples. Thus the reviewer suggests conducting experiments using dynamic evaluation instead of static evaluation. 

[1] Chenglei Si, Zhengyan Zhang, Fanchao Qi, Zhiyuan Liu, Yasheng Wang, Qun Liu, and Maosong Sun. 2021. Better Robustness by More Coverage: Adversarial and Mixup Data Augmentation for Robust Finetuning. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 1569–1576, Online. Association for Computational Linguistics.

>> We acknowledge the concern of reviewer about more comprehensive perturbation strategies and thank them for their valuable feedback, however, for the purpose of this study and only to achieve generalizable perturbations that can solely show the effect of perturbations on language models, we use three different models as target models when gathering the perturbations to ensure generalizability of our perturbations and this is based on common practice (done by the authors of ADVGlue as well). 

---

3. Missing comparision with basic robustness baselines, e.g. adversarial training.

>> The main point of this paper is to utilize prototype-based networks and check if their inherent robustness properties will be transferred to text classification tasks as well, and their comparison with language models that do not utilize prototypes. We did not intend to achieve robustness by adversarial training, and that's why we do not include it in our experiments.

---

4. As a critical point, the reviewer remains unconvinced that solely prototype can lead to adversarial robustness. Beside, the authors frequently highlight the interpretability of PBNs, but there is a lack of analysis or experimental evidence supporting this claim. 

>>We thank the reviewer for pointing out this important aspect of our study. The classification in PBNs is done based on the distance of examples to prototypes, and we claim that they have two advantages: 1. Since the model is only allowed to leverage the distances to prototypes, perturbations have less direct effect on the outcome of the prediction, which otherwise, would’ve been a more direct effect on the embedding space; 2. Distances as suggested by the vision community can help identify perturbations even if recovering the correct prediction is difficult which help the model/user understand if the model is facing a perturbed example. We plan to add a comprehensive discussion of interpretability both in terms of what prototypes represent (Our experiments show that prototypes capture prominent semantic features of the dataset) and how they change by perturbations (utilizing PBNs, the change that happens in the prediction outcome is reflected in the change in prototypes and model keeps its interpretability properties even under effective perturbations) that are completely ready.




## Questions For The Authors:
>>The questions provided are really good and we are really thankful for such constructive feedback. Below we provide answers for the raised questions. For the camera ready version, we plan to address these points directly in the discussion of the results to make our contribution stronger.

1. The validation set in AdvGLUE SST-2 consists of 148 examples, but in the experiment, only the first 131 examples were used. The reason behind this choice is not clear and requires further explanation.

>>We thank the reviewer for their thorough review and mentioning this point. Upon requesting the validation set of AdvGLUE SST-2 from the authors that would contain type of perturbations, 17 examples were found to be human-generated and contained mixed perturbation types. The human annotations do not have original sentences because the sentences were created specifically as an attack and are not manipulations of the original text; hence we discarded those data points, proceeding with the left 131 examples. 

---

2. Why were static adversarial examples generated to attack BERT family models, but PBNs were not tested on BERT? Clarification on this point is needed.

>>We used BERT, RoBERTa, and DistilBERT to generate the perturbations that are generalizable enough (perturbations can change three different models' predictions) that could represent perturbations effect on a variety of models. We did not use BERT-base as a backbone for prototype-based reasoning models since we used BERT-base models as the target model for our perturbations. This separation ensures that our evaluation framework is generalizable and doesn't put PBNs nor their backbones in a direct targeted spot by perturbations. Nevertheless, for the sake of completeness of our experiments, we conducted additional experiments using BERT as backbone, which their performance aligned with the results using other backbones reported in the paper. We plan to include this additional results in the paper. 


|           |     IMDB |          |          |          |  AG News |          |          |          |  DBPedia |          |          |          |    SST-2 |          |
|----------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| **Model** | **Orig** | **Char** | **Word** | **Sent** | **Orig** | **Char** | **Word** | **Sent** | **Orig** | **Char** | **Word** | **Sent** | **Orig** |  **Adv** |
|    BERT-S |     96.4 |     84.6 | **86.4** | **91.3** |     94.4 |     73.4 | **79.9** | **88.1** |     97.9 |     58.8 |     57.9 | **97.8** |     84.7 |     40.7 |
|    \+ PBN |     91.0 | **84.7** |     86.0 |     87.3 |     91.9 | **74.5** |     79.0 |     84.9 |     98.9 | **67.4** | **73.5** |     97.0 |     76.2 | **44.0** |
|       *Δ* |     -5.4 |     +0.1 |     -0.4 |     -4.0 |     -2.5 |     +1.1 |     -0.9 |     -3.2 |     +1.0 |     +8.6 |    +15.6 |     -0.8 |     -8.5 |     +3.3 |
|    BERT-M |     94.7 | **84.2** |     85.5 | **92.2** |     93.9 |     71.1 |     78.8 |     88.3 |     98.4 |     66.2 |     60.5 | **98.0** |     83.9 |     40.9 |
|    \+ PBN |     90.5 |     83.5 | **85.6** |     85.9 |     92.1 | **78.4** | **80.2** | **88.5** |     96.5 | **69.0** | **75.5** |     97.4 |     77.8 | **46.3** |
|       *Δ* |     -4.2 |     -0.7 |     +0.1 |     -6.3 |     -1.8 |     +7.3 |     +1.4 |     +0.2 |     -1.9 |     +2.8 |    +15.0 |     -0.6 |     -6.1 |     +5.4 |
---

Missing References:

NA

Typos Grammar Style And Presentation Improvements:

Misleading expression:

1. The frequent reference to LLMs is inappropriate. As indicated in line 063, (Moradi and Samwald, 2021) only addressed the robustness problem in small pretrained models (parameters less than 10B). Similarly, in line 064, the phrase "Addressing the shortcomings of LLMs" is also misleading, as this paper specifically focuses on small pretrained models.

>>We thank the reviewer for pointing us to this issue. References to LLMs will be corrected where necessary and the distinction between pre-trained language models and large language models will be made in the paper.

---

2. The default TextBugger in TextAttack is not solely a character-level attack. It comprises both character-level and word-level perturbations [1].

[1]https://github.com/QData/TextAttack/blob/00adb8a55580f6dea5fd6952e93f095829e807dd/textattack/attack_recipes/textbugger_li_2018.py#L74C53-L74C53

>>We do not use the WordSwapEmbedding in the textBugger implementation, However, we will check all the perturbations again to remove any potential doubts.


--- 

Soundness: 2: Borderline: Some of the main claims/arguments are not sufficiently supported, there are major technical/methodological problems

Excitement: 3: Ambivalent: It has merits (e.g., it reports state-of-the-art results, the idea is nice), but there are key weaknesses (e.g., it describes incremental work), and it can significantly benefit from another round of revision. However, I won't object to accepting it if my co-reviewers champion it.

Reproducibility: 2: Would be hard pressed to reproduce the results. The contribution depends on data that are simply not available outside the author's institution or consortium; not enough details are provided.

Ethical Concerns: No

Reviewer Confidence: 4: Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.

