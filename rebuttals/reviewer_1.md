# REVIEW 1

>>We thank the reviewers for their feedback and insightful suggestions. We are pleased that the paper idea is overall positively perceived by most of the reviewers. We now clarify the questions raised by the reviewers.

## Paper Topic And Main Contributions:
This paper analyzes the robustness of an interesting family of models: Prototype-Based Networks (PBNs). The authors study the robustness of PBNs against adversarial attacks of char/word/sentence levels in terms of Backbones, Distance functions, Number of prototypes, and Objective functions. Experimental results on four sentence classification tasks show that PBNs are more robust to adversarial attacks than their respective vanilla counterparts. The authors also give empirical results on impacts of backbones and other implementation details.


## Reasons To Accept:
1. This paper studies an interesting problem: the robustness of prototype networks. PBNs are rather not popular in NLP. Understanding how such models work under adversarial conditions brings new insights to the community.
2. Experimental results are thorough, providing insights on impacts of backbone types, sizes, loss functions, distance functions as well as number of prototypes.


## Reasons To Reject:
1. Figure 1 is rather confusing for audience who has no prior knowledge of what PBNs are. And the introduction from line 118 to line 156 also does not fully explains what is a PBN clearly. The authors should consider either improve Figure 1 or more line 118 to 156 more clearly by adding a few equations.

>>We will improve the figure as well as the corresponding explanation on PBNs. We attach our revised explanation here for your reference: 

>>_"PBNs are a family of interpretable models that classify data points based on their similarity to certain prototypes developed during training. Learned prototypes summarize the prominent semantic patterns of the dataset, and being close or far from these semantic patterns yields the prediction outcome in PBNs. This property is achieved through two mechanisms: 1. prototypes are defined in the same embedding space as input examples, which makes them interpretable by leveraging input examples close to them; 2. prototypes are designed to cluster semantically similar training examples, which makes them representative of the prominent patterns embedded in the dataset and input examples. The decisions made by PBNs are inherently interpretable because prototypes are trained to be aligned with previous observations. This property helps users to understand the behavior of the model during inference better by looking at the closest activated prototypes."_

---

2. Evaluation details missing. To study the size of Transformer models on PBNs, the authors include two models, BERT-Small, and BERT-Medium. While standard BERT-base model has 110M params, it is not clear how the authors obtain BERT-S and BERT-M. Same applies to BART (see table 4), standard BART-base has around 220M params, not clear of how the authors obtain BART of around 50M params. Besides, Figure 2 evaluates two models, BART 1x and BART 2x, it is also not clear of how these two models are obtained, and the details of the number of params for these two models are also missing.

>>We thank the reviewer for mentioning this point. We plan to add all the mentioned details and other related nuances of our experimental setup in the revised version of the paper. Models used as target to gather generalizable perturbations were BERT-base, RoBERTa-base, and DistilBERT-base fine-tuned on three datasets separately for getting the perturbations for each dataset. BERT models used as baselines were BERT-medium [1] and BERT-small [2] that are the smaller pre-trained BERT variants chosen with respect to having a closer number of parameters to PBN variants analyzed in our study. 

>>We only used the encoder part of BART architecture for our experiments and that’s the reason behind it being smaller than usual BART models. The BART and ELECTRA model used in our main experiments were BART-base and ELECTRA-base and the bigger variant of BART that we used to test the effect of scale is BART-large encoder. In terms of pre-training, the models used as target models to gather generalizable perturbations were fine-tuned on three datasets separately for getting the perturbations for each dataset. PBN models and their backbone were all fine-tuned on each dataset separately (backbones were not pre-trained further than their original pre-training).

>>[1] https://huggingface.co/prajjwal1/bert-medium

>>[2] https://huggingface.co/prajjwal1/bert-small 

---

3. Experimental setup is confusing. To study the scaling of transformer backbones, the authors studied BART 1x and BART 2x in Figure2, why study these two while you already have results on BERT-S and BERT-M? Results of BERT-base and BERT-large can be obtained very easily too.

>>The study of scaling up the transformer backbones was intended to test their effectiveness when used as a backbone for PBNs, hence we used BART-base and BART-large as PBN’s backbones for this experiment. BERT-small and BERT-medium were only used as baselines to incorporate models that are from the BERT family but closer to PBN models in terms of number of parameters for a more fair comparison.

---

4. Line 476 to line 497 ablates the effect of different loss terms, while Figure 3 shows that clustering loss brings negative effect on the performance, line 476 to line 497 does not provide any explanations, and it seems that throughout the paper the authors did activate the clustering loss, leading to a sub-optimal experimental setting.


>>The goal of this study is to provide a framework for evaluating the robustness of PBNs by incorporating common design choices in literature, and not necessarily achieving the highest robustness. Furthermore, all the objective functions included in the framework are proven to be essential for interpretability of PBNs that if otherwise discarded eliminate any reason for using PBNs. On this ground, we only consider eliminating objective functions and test their effect on robustness in their own regard and their own ablation study. 



## Questions For The Authors:

>>The questions provided are really good and we are really thankful for such constructive feedback. Below we provide answers for the raised questions. For the camera ready version, we plan to address these points directly in the discussion of the results to make our contribution stronger.

For soundness of the paper, I am mainly confused on the non-standard models the authors evaluated in the paper. How are these models pretrained and what are the details of the architecture (BART 1x and 2x in Figure2, BERT-S and BERT-M, BART in table 4, etc)? 

>> We thank the reviewer for mentioning this point. We plan to add all the mentioned details and other related nuances of our experimental setup in the revised version of the paper. Models used as target to gather generalizable perturbations were BERT-base, RoBERTa-base, and DistilBERT-base fine-tuned on three datasets separately for getting the perturbations for each dataset. BERT models used as baselines were BERT-medium [1] and BERT-small [2] that are the smaller pre-trained BERT variants chosen with respect to having a closer number of parameters to PBN variants analyzed in our study. 

>> We only used the encoder part of BART architecture for our experiments and that’s the reason behind it being smaller than usual BART models. The BART and ELECTRA model used in our main experiments were BART-base and ELECTRA-base and the bigger variant of BART that we used to test the effect of scale is BART-large encoder. In terms of pre-training, the models used as target models to gather generalizable perturbations were fine-tuned on three datasets separately for getting the perturbations for each dataset. PBN models and their backbone were all fine-tuned on each dataset separately (backbones were not pre-trained further than their original pre-training).

>> [1] https://huggingface.co/prajjwal1/bert-medium

>> [2] https://huggingface.co/prajjwal1/bert-small 

---

Soundness: 2: Borderline: Some of the main claims/arguments are not sufficiently supported, there are major technical/methodological problems

Excitement: 3: Ambivalent: It has merits (e.g., it reports state-of-the-art results, the idea is nice), but there are key weaknesses (e.g., it describes incremental work), and it can significantly benefit from another round of revision. However, I won't object to accepting it if my co-reviewers champion it.

Reproducibility: 4: Could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.

Ethical Concerns: No

Reviewer Confidence: 4: Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
