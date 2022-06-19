# Introduction

In this project, I experiment on Aspect-based Sentiment Analysis (ABSA) task by fine-tuning some state-of-the-art pretrained language models including [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805), [XLNet (Yang et al., 2019)](https://arxiv.org/abs/1906.08237), [RoBERTa (Liu et al., 2019)](https://arxiv.org/abs/1907.11692) and [XLM-R (Conneau et al., 2020)](https://arxiv.org/abs/1911.02116). This task gets a comment and an aspect as input, and return the sentiment that comment expresses about that aspect. For example, with the comment "*But the staff was so horrible to us*" and the aspect "*Service*", the corresponding sentiment is "*Negative*".

Then I improve the results by using emsemble learning technique including voting, weighted averaging and blending.

# Dataset

I did my experiment on [SemEval 2014 Task 4 Restaurant](https://alt.qcri.org/semeval2014/task4/) (Res14) and [SemEval 2016 Task 5 Restaurant](https://alt.qcri.org/semeval2016/task5/) (Res16). These are two benchmark datasets for ABSA task.

# Quickstart

Firstly, you need to install all the required dependencies:

```
pip install -r requirements.txt
```

Download the weight files I trained before:

```
bash download.sh
```

To get the report about results of all models for each dataset, run this command:

```
python evaluate.py --dataset [res14 or res16]
```

# How to train

```
python train.py \
--model_name [bert, roberta, xlnet or xlmr] \
--dataset [res14 or res16] \
--optimizer [adam, sgd, adagrad or rmsprop] \
--lr 1e-5 \
--dropout 0.1 \
--num_epoch 30 \
--batch_size 64 \
--log_step 10 \
--pretrained_dim 768 \
--pretrained_name [for bert: bert-base-uncased] \
--max_seq_len 50 \
--polarities_dim 3 \
--device [(optional) cpu or gpu]
```

After training all 4 models for each dataset, you can train meta-model for Blending:

```
python train_meta_mlp.py \
--dataset [res14 or res16] \
--optimizer [adam, sgd, adagrad or rmsprop] \
--lr 1e-5 \
--num_epoch 100 \
--batch_size 64 \
--log_step 10 \
--hidden_dim 100 \
--polarities_dim 3 \
--device [(optional) cpu or gpu]
```

# Results
|             | Res14 | Res16 |
| ------------------ | ------------------ | ------------------ |
| BERT | 88.49/80.29 | 86.28/70.98 |
| XLNet | 91.26/84.54 | 88.02/72.62 |
| ROBERTa | 90.96/83.56 | 88.55/76.61 |
| XLM-R | 88.59/79.96 | 86.95/68.29 |
| Hard-voting | 91.78/84.34 | 88.68/71.29 |
| Soft-voting | 92.29/85.61 | 89.35/75.48 |
| Weighted Averaging | 92.39/85.91 | 89.35/75.48 |
| Blending | 92.70/86.46 | 89.75/75.03 |

# Acknowledgement

I want to thank Dr. Bui Khac Hoai Nam for supporting me in this project.