# Bias-and-Fairness-in-Large-Language-Models

## Introduction
This study is about to reproduce a recent ACL paper which is titled Bias and fairness in large language models: A survey (Gallegos et al., 2024[1]). We aim to verify the key findings and experimental results through our own replication efforts, which includes two questions: Do LLMs exhibit inherent biases? How can we correct such biases?

### Research Background
Why do we want to analyze the bias and fairness in LLMs? The rise and rapid advancement of large language models (LLMs) has fundamentally changed language technologies (e.g., Brown et al 2020[2]). Laying behind the change, however, is the potential to perpetuate harm. Typically trained on an enormous scale of uncurated Internet - based data, LLMs inherit stereotypes, misrepresentations, derogatory and exclusionary language, and other degrading behaviors that disproportionately affect already - vulnerable and marginalized communities (Bender et al. 2021[3]). These harms are forms of “social bias”, which we broadly use to refer to disparate treatment or outcomes between social groups that arise from historical and structural power asymmetries. Thus, it is important and meaningful for us to test and correct the biases of LLMs. 

### Research Framework
How can we correct such biases? For this question, we firstly reproduce the debias techniques in Gallegos et al (2024), which includes Prompt Fine-tuning and Supervised Fine-tuning. Prompt Fine-tuning entails crafting prompts with elements like bias label explanations (e.g., left/right-leaning perspectives from Wikipedia), or debiasing statements (e.g., using “Please ensure that your answer is unbiased” as prompt). Supervised Fine-tuning refers to the process of optimizing a pre-trained model using labeled downstream task data (e.g., question-answer pairs) to update its parameters for specific tasks.

What’s more, we then refer to Lin et al. (2024)[4], which is a comprehensive review. Based on this paper, we find interest in Loss Function Modification for debiasing, which is to change the loss function based on Embedding, Attention or Predicted Token Distribution. We mostly modify the loss function for word embedding debiasing according to Yang et al. (2025)[5] and Kaneko and Bollegala (2021)[6], which is to address bias in the hidden representations of an encoder via a new equalizing objective or regularization constraints. We firstly use the dataset about gender bias and obtain some ideal results, then try this idea on the dataset about political bias.

## References
[1]	Lin, Luyang, Lingzhi Wang, Jinsong Guo, and Kam-Fai Wong. 2024. Investigating Bias in LLM-Based Bias Detection: Disparities between LLMs and Human Perception. arXiv preprint arXiv:2403.14896.

[2]	Brown, Tom, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few - shot learners. Advances in Neural Information Processing Systems, 33:1877–1901.

[3]	Bender, Emily M. 2019. A typology of ethical risks in language technology with an eye towards where transparent documentation can help. Presented at The Future of Artificial Intelligence: Language, Ethics, Technology Workshop.

[4]	Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. 2024. Bias and Fairness in Large Language Models: A Survey. Computational Linguistics, 50(3):1097–1179.

[5]	Yang, Ke, Charles Yu, Yi Fung, Manling Li, and Heng Ji. 2025. ADEPT: A DEbiasing PrompT Framework. arXiv preprint arXiv:2211.05414v3.

[6]	Kaneko, M.; and Bollegala, D. 2021. Debiasing pre-trained contextualised embeddings. arXiv preprint arXiv:2101.09523.
