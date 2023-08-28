# REVIEW 2

>>We thank the reviewers for their feedback and insightful suggestions. We are pleased that the paper idea is overall positively perceived by most of the reviewers. We now clarify the questions raised by the reviewers.

## Paper Topic And Main Contributions:

This paper delves into an in-depth investigation of the robustness of prototype-based networks (PBN) in text classification tasks. Through extensive experimentation, the study explores the impact of various aspects, including backbone architectures, distance functions, number of prototypes, and objective functions, under different levels of perturbations (character-level, word-level, and sentence-level). The results of the experiments, conducted on four datasets, shed light on the crucial role played by the objective function in enhancing the robustness of PBNs, thus making prototypes more interpretable. Additionally, when compared to non-PBN structures on the transformer-based architecture, PBNs demonstrate higher levels of robustness. Furthermore, the study concludes that the Euclidean distance outperforms the Cosine distance in the context of PBNs. It is observed that robust PBNs necessitate a greater number of prototypes than the number of classes.


## Reasons To Accept:
1. Previous research on PBNs has predominantly centered around exploring their robustness in image classification tasks. However, this paper stands out by delving into the analysis of how various factors affect the robustness of PBNs in text classification tasks.

2. The study carried out an extensive and thorough analysis, comprehensively investigating the robustness of PBNs from four distinct perspectives. Additionally, the authors explored the influence of three different levels of perturbations on the model's overall robustness.

## Reasons To Reject:
1. One notable weakness of the paper lies in its limited analysis, as it primarily relies on experimental results without offering reasonable explanations for the observed phenomena. The fundamental principles guiding the experiments were not adequately elucidated, leaving important questions unanswered. For instance, the paper does not provide a clear rationale for why the absence of clustering loss leads to improved performance on the datasets of IMDB and AG News, nor does it explain the reasons behind the superior performance of Euclidean distance over cosine distance. Moreover, the significant performance disparity between PBNs with transformer-based backbones and CNNs remains unexplained, leaving a gap in our understanding.


>>Thorough explanations are provided in the section “Questions For The Authors”.

--- 

2. Another area of concern is the insufficient justification for certain experimental settings. Although the main goal of the paper is to explore the robustness of PBNs in text classification tasks, the experiments concerning the influence of backbone structures and parameter selection were conducted solely on BERT under different perturbations. The absence of an experiment on PBNs with BERT as the backbone under different perturbations raises questions about the comprehensiveness of the study. It is essential to address this limitation as it might deviate from the core claim of the paper and might not offer a holistic understanding of PBNs' robustness in various scenarios. Including additional experiments with PBNs and BERT as the backbone under different perturbations could bolster the paper's credibility and strengthen its conclusions.

>>We used BERT, RoBERTa, and DistilBERT to generate the perturbations that are generalizable enough (perturbations can change three different models' predictions) that could represent perturbations effect on a variety of models. We did not use BERT-base as a backbone for prototype-based reasoning models since we used BERT-base models as the target model for our perturbations. This separation ensures that our evaluation framework is generalizable and doesn't put PBNs nor their backbones in a direct targeted spot by perturbations. Nevertheless, for the sake of completeness of our experiments, we conducted additional experiments using BERT as backbone, which their performance aligned with the results using other backbones reported in the paper. We plan to include this additional results in the paper.


## Questions For The Authors:

>>The questions provided are really good and we are really thankful for such constructive feedback. Below we provide answers for the raised questions. For the camera ready version, we plan to address these points directly in the discussion of the results to make our contribution stronger.

---

1. Can you provide a comprehensive explanation for the observed improvement in performance on specific tasks, such as IMDB and AG News, when the clustering loss is absent?


>> The clustering loss is an additional regularization term in the overall loss function. As usual with regularization terms, the term forces the network to learn a trade-off between the competing goals. In our case, all loss terms, including the accuracy target loss, are somewhat competing goals. Therefore, the absence of a regularization term could cause an improvement of the other loss terms if the goals were competing. The clustering loss forces the backbone to project all samples to be close to a prototype in the embedding space; An objective that is useful for interpretability because it ensures that the closest prototype serves as a valid proxy for the sample. However, this loss lowers the possible achievable accuracy because forcing each sample to be close to a prototype lowers the diversity in the embedding space as our observations suggest as well (0.9 compared to 1.0).

---

2. What is the underlying rationale behind the selection of 16 as the optimal number of prototypes, and why does the model's performance deteriorate as the number of prototypes increases?

>>During training the optimal number of prototypes was automatically selected. From literature [1], it is known that the optimal number of prototypes is non-trivial and not necessarily the number of classes or training data points. Therefore, the number 16 is in accordance with results already published.

>>[1] Crammer, K., Gilad-Bachrach, R., Navot, A., & Tishby, N. (2002). Margin analysis of the LVQ algorithm. Advances in neural information processing systems, 15.

---

3. Could you elucidate the reasons behind the superior performance obtained by using the Euclidean distance metric compared to the cosine distance metric?

>>In prototype-based learning an appropriate selection of the dissimilarity measure is one of the major investigation subjects [1, 2]. Thus, the dissimilarity selection is usually treated as a hyperparameter. In our case, the Euclidean distance works better than the cosine similarity. However, this doesn’t imply that this holds always. One reason for the different performance might be that the Euclidean distance forces the embedded samples to be close to a point (the prototype) in the vector space whereas the cosine similarity forces embedded samples to be in the same direction as the prototype. Consequently, the formation of clusters is completely different with the two measures leading to the observed performance differences.

>>[1] Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019). This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems, 32.

>>[2] Mettes, P., Van der Pol, E., & Snoek, C. (2019). Hyperspherical prototype networks. Advances in neural information processing systems, 32.

---

Missing References:

N/A

Typos Grammar Style And Presentation Improvements:

N/A

Soundness: 3: Good: This study provides sufficient support for its major claims/arguments, some minor points may need extra support or details.

Excitement: 2: Mediocre: This paper makes marginal contributions (vs non-contemporaneous work), so I would rather not see it in the conference.

Reproducibility: 4: Could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.

Ethical Concerns: No

Justification For Ethical Concerns:

N/A

Reviewer Confidence: 4: Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
