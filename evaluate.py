import logging
import argparse
import sys
import time
import os

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn import metrics
from time import strftime, localtime
from scipy.special import softmax

import torch
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer, XLMRobertaTokenizer

from data_utils import ABSADataset, ABSATestDataset
from models import BertClassifier, RobertaClassifier, XLNetClassifier, XLMRobertaClassifier, MLPClassifier, LinearClassifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def hard_voting(all_pred, output, logger, cfm_folder, dataset):
    all_pred = np.argmax(all_pred, axis=-1)
    all_pred = all_pred.T
    meta_pred = []
    for preds in all_pred:
        count = np.array([0, 0, 0])
        for pred in preds:
            count[pred] += 1
        pred = np.argmax(count, axis=0)
        meta_pred.append(pred)
    meta_pred = np.array(meta_pred)
    
    logger.info(f'Hard-voting: accuracy {metrics.accuracy_score(output, meta_pred):.4f} f1 {metrics.f1_score(output, meta_pred, average="macro"):.4f} \n')
    plot_cf_matrix(output, meta_pred, cfm_folder, dataset, 'hard-voting')

def soft_voting(all_pred, output, logger, cfm_folder, dataset):
    all_pred = softmax(all_pred, axis=-1)
    all_pred = np.transpose(all_pred, (1, 2, 0))
    all_pred = np.mean(all_pred, axis=-1)
    all_pred = np.argmax(all_pred, axis=-1)
    meta_pred = all_pred
    
    logger.info(f'Soft-voting: accuracy {metrics.accuracy_score(output, meta_pred):.4f} f1 {metrics.f1_score(output, meta_pred, average="macro"):.4f} \n')
    plot_cf_matrix(output, meta_pred, cfm_folder, dataset, 'soft-voting')

def weighted_averaging(all_pred, output, logger, cfm_folder, dataset):
    all_pred = softmax(all_pred, axis=-1)
    all_pred = np.transpose(all_pred, (1, 2, 0))

    if dataset == 'res14':
        weight = np.array([0.7511, 0.7743, 0.7825, 0.7525])
    else:
        weight = np.array([0.7933, 0.7673, 0.7628, 0.7532])
    
    all_pred = np.average(all_pred, weights=weight, axis=-1)
    all_pred = np.argmax(all_pred, axis=-1)
    meta_pred = all_pred

    logger.info(f'Weighted Averaging: accuracy {metrics.accuracy_score(output, meta_pred):.4f} f1 {metrics.f1_score(output, meta_pred, average="macro"):.4f} \n')
    plot_cf_matrix(output, meta_pred, cfm_folder, dataset, 'weighted_averaging')

def mlp_blending(all_pred, output, logger, weight_folder, hidden_dim, polarities_dim, cfm_folder, dataset, device):
    all_pred = np.transpose(all_pred, (1, 2, 0))
    all_pred = np.reshape(all_pred, (all_pred.shape[0], -1))
    all_pred = torch.tensor(all_pred, dtype=torch.float).to(device)
    
    meta_model = MLPClassifier(all_pred.shape[-1], hidden_dim, polarities_dim)
    meta_model.load_state_dict(torch.load(os.path.join(weight_folder, dataset, f'meta-{args.dataset}-blending-mlp.pth'), map_location=device))
    meta_model.to(device)

    with torch.no_grad():
        meta_pred = meta_model(all_pred)
        meta_pred_ = meta_pred.argmax(dim=1).cpu().detach().numpy()

        logger.info(f'MLP Blending: accuracy {metrics.accuracy_score(output, meta_pred_):.4f} f1 {metrics.f1_score(output, meta_pred_, average="macro"):.4f} \n')

    plot_cf_matrix(output, meta_pred_, cfm_folder, dataset, 'mlp_blending')

def linear_blending(all_pred, output, logger, weight_folder, cfm_folder, dataset, device):
    all_pred = np.transpose(all_pred, (1, 2, 0))
    all_pred = torch.tensor(all_pred, dtype=torch.float).to(device)

    meta_model = LinearClassifier(all_pred.shape[-1])
    meta_model.load_state_dict(torch.load(os.path.join(weight_folder, dataset, f'meta-{args.dataset}-blending-linear.pth'), map_location=device))
    meta_model.to(device)

    with torch.no_grad():
        meta_pred = meta_model(all_pred)
        meta_pred_ = meta_pred.argmax(dim=1).cpu().detach().numpy()

        logger.info(f'Linear Blending: accuracy {metrics.accuracy_score(output, meta_pred_):.4f} f1 {metrics.f1_score(output, meta_pred_, average="macro"):.4f} \n')

    plot_cf_matrix(output, meta_pred_, cfm_folder, dataset, 'linear_blending')

def plot_cf_matrix(output, pred, cfm_folder, dataset, method):
    cf_matrix = metrics.confusion_matrix(output, pred)
    heatmap = sns.heatmap(cf_matrix, annot=True, xticklabels=['Positive', 'Negative', 'Neutral'], yticklabels=['Positive', 'Negative', 'Neutral'], fmt='d')
    heatmap.set(xlabel='Predicted label', ylabel='True label')
    fig = heatmap.get_figure()
    fig.savefig(os.path.join(cfm_folder, dataset, method+'.png'), dpi=300)
    plt.clf()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--method', default='all', type=str)
    parser.add_argument('--dataset', default='res14', type=str, help='res14 or res16')

    args = parser.parse_args()

    log_folder = 'log'
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if not os.path.exists(os.path.join(log_folder, args.dataset)):
        os.mkdir(os.path.join(log_folder, args.dataset))
    log_file = f'{args.method}-{args.dataset}-{strftime("%d%m%y-%H%M", localtime(time.time()+7*3600))}.log'
    logger.addHandler(logging.FileHandler(os.path.join(log_folder, log_file)))

    logger.info(f'- Method: {args.method}')
    logger.info(f'- Dataset: {args.dataset}\n')
    
    dataset_files = {
        'res14': {
            'train': 'data/res14/Restaurants_Train.xml',
            'test': 'data/res14/Restaurants_Test.xml'
        },
        'res16': {
            'train': 'data/res16/ABSA16_Restaurants_Train_SB1_v2.xml',
            'test': 'data/res16/EN_REST_SB1_TEST.xml.gold'
        }
    }
    
    pretrained_dim = 768
    polarities_dim = 3
    dropout = 0.1
    max_seq_len = 50
    hidden_dim = 100

    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    
    weight_folder = 'weight'
    if not os.path.exists(weight_folder):
        os.mkdir(weight_folder)
    if not os.path.exists(os.path.join(weight_folder, args.dataset)):
        os.mkdir(os.path.join(weight_folder, args.dataset))
    
    cfm_folder = 'cfm'
    if not os.path.exists(cfm_folder):
        os.mkdir(cfm_folder)
    if not os.path.exists(os.path.join(cfm_folder, args.dataset)):
        os.mkdir(os.path.join(cfm_folder, args.dataset))

    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    roberta_model = RobertaClassifier('roberta-base', pretrained_dim, polarities_dim, dropout)
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    bert_model = BertClassifier('bert-base-uncased', pretrained_dim, polarities_dim, dropout)
    
    xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    xlnet_model = XLNetClassifier('xlnet-base-cased', pretrained_dim, polarities_dim, dropout)
    
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=True)
    xlmr_model = XLMRobertaClassifier('xlm-roberta-base', pretrained_dim, polarities_dim, dropout)

    num_model = 4

    tokenizers = [roberta_tokenizer, bert_tokenizer, xlnet_tokenizer, xlmr_tokenizer]
    models = [roberta_model, bert_model, xlnet_model, xlmr_model]
    model_names = ['roberta', 'bert', 'xlnet', 'xlmr']

    all_pred = []

    for i in range(num_model):
        logger.info(f'{model_names[i]}')
        test_data = ABSATestDataset(args.dataset, dataset_files[args.dataset]['test'], tokenizers[i], model_names[i], max_seq_len)

        test_label = torch.tensor(test_data.data['label']).to(device)

        test_input_ids = torch.tensor(test_data.data['input_ids']).to(device)
        test_token_type_ids = torch.tensor(test_data.data['token_type_ids']).to(device)
        test_attention_mask = torch.tensor(test_data.data['attention_mask']).to(device)

        models[i].load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'{model_names[i]}-{args.dataset}.pth')))
        models[i].to(device)
        logger.info(f'Restore {model_names[i]} model !')
    
        with torch.no_grad():
            pred = models[i](input_ids=test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids)
            pred_ = pred.argmax(dim=1).cpu().detach().numpy()
            
            output = test_label.cpu().detach().numpy()
            logger.info(f'Test: accuracy {metrics.accuracy_score(output, pred_):.4f} f1 {metrics.f1_score(output, pred_, average="macro"):.4f} \n')
            
            plot_cf_matrix(output, pred_, cfm_folder, args.dataset, model_names[i])
            all_pred.append(pred.tolist())
    
    all_pred = np.array(all_pred)
    
    if args.method == 'all':
        hard_voting(all_pred, output, logger, cfm_folder, args.dataset)
        soft_voting(all_pred, output, logger, cfm_folder, args.dataset)
        weighted_averaging(all_pred, output, logger, cfm_folder, args.dataset)
        mlp_blending(all_pred, output, logger, weight_folder, hidden_dim, polarities_dim, cfm_folder, args.dataset, device)
        linear_blending(all_pred, output, logger, weight_folder, cfm_folder, args.dataset, device)
    
    elif args.method == 'hard-voting':
        hard_voting(all_pred, output, logger, cfm_folder, args.dataset)
    
    elif args.method == 'soft-voting':
        soft_voting(all_pred, output, logger, cfm_folder, args.dataset)
    
    elif args.method == 'weighted_averaging':
        weighted_averaging(all_pred, output, logger, cfm_folder, args.dataset)
    
    else:
        mlp_blending(all_pred, output, logger, weight_folder, hidden_dim, polarities_dim, cfm_folder, args.dataset, device)
        

                