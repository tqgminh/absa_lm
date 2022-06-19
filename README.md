# Introduction

In this project, I experiment on Aspect-based Sentiment Analysis (ABSA) task by fine-tuning some state-of-the-art pretrained language models including [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805), [XLNet (Yang et al., 2019)](https://arxiv.org/abs/1906.08237), [RoBERTa (Liu et al., 2019)](https://arxiv.org/abs/1907.11692) and [XLM-R (Conneau et al., 2020)](https://arxiv.org/abs/1911.02116). This task gets a comment and an aspect as input, and return the sentiment that comment expresses about that aspect. For example, with the comment "*But the staff was so horrible to us*" and the aspect "*Service*", the corresponding sentiment is "*Negative*".

# Dataset

I did my experiment on [SemEval 2014 Task 4 Restaurant](https://alt.qcri.org/semeval2014/task4/) (Res14) and [SemEval 2016 Task 5 Restaurant](https://alt.qcri.org/semeval2016/task5/) (Res16). These are two benchmark datasets for ABSA task.

# Quickstart

Firstly, you need to install all the required dependencies:

```
pip install -r requirements.txt
```
