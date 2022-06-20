import argparse
import time
import os

import numpy as np
from scipy.special import softmax

import torch
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer, XLMRobertaTokenizer

from data_utils import ABSADataset, pad_and_truncate
from models import BertClassifier, RobertaClassifier, XLNetClassifier, XLMRobertaClassifier, MLPClassifier

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--method', default='bert', type=str, help='bert, roberta or xlnet')
    parser.add_argument('--dataset', default='res14', type=str, help='res14 or res16')
    parser.add_argument('--input', type=str)

    args = parser.parse_args()
    
    r = open(args.input, 'r')
    content = r.readlines()
    sentence = content[0][:-1]
    print(sentence)
    
    aspect = content[1].replace('/', ' ').lower().replace('#', ' ').replace('_', ' ')
    print(aspect)
    r.close()

    max_seq_len = 50
    pretrained_dim = 768
    polarities_dim = 3
    dropout = 0.1
    hidden_dim = 100

    weight_folder = 'weight'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.method == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        model = RobertaClassifier('roberta-base', pretrained_dim, polarities_dim, dropout)
        
        encoding = tokenizer.encode_plus(aspect, sentence, return_token_type_ids=True)
        tmp = {}
        tmp['input_ids'] = [pad_and_truncate(encoding['input_ids'], max_seq_len).tolist()]
        tmp['token_type_ids'] = [pad_and_truncate(encoding['token_type_ids'], max_seq_len).tolist()]
        tmp['attention_mask'] = [pad_and_truncate(encoding['attention_mask'], max_seq_len).tolist()]

        test_input_ids = torch.tensor(tmp['input_ids']).to(device)
        test_token_type_ids = torch.tensor(tmp['token_type_ids']).to(device)
        test_attention_mask = torch.tensor(tmp['attention_mask']).to(device)

        model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'{args.method}-{args.dataset}.pth'), map_location=device))
        model.to(device)
        
        start = time.time()
        with torch.no_grad():
            pred = model(input_ids=test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids)
            pred_ = pred.argmax(dim=1).cpu().detach().numpy()

            if pred_[0] == 0:
                print('Positive')
            elif pred_[0] == 1:
                print('Negative')
            else:
                print('Neutral')
        print(f'Inference time: {time.time() - start:.2f}')

    elif args.method == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model = BertClassifier('bert-base-uncased', pretrained_dim, polarities_dim, dropout)

        encoding = tokenizer.encode_plus(aspect, sentence, return_token_type_ids=True)
        tmp = {}
        tmp['input_ids'] = [pad_and_truncate(encoding['input_ids'], max_seq_len).tolist()]
        tmp['token_type_ids'] = [pad_and_truncate(encoding['token_type_ids'], max_seq_len).tolist()]
        tmp['attention_mask'] = [pad_and_truncate(encoding['attention_mask'], max_seq_len).tolist()]

        test_input_ids = torch.tensor(tmp['input_ids']).to(device)
        test_token_type_ids = torch.tensor(tmp['token_type_ids']).to(device)
        test_attention_mask = torch.tensor(tmp['attention_mask']).to(device)

        model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'{args.method}-{args.dataset}.pth'), map_location=device))
        model.to(device)
        
        start = time.time()
        with torch.no_grad():
            pred = model(input_ids=test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids)
            pred_ = pred.argmax(dim=1).cpu().detach().numpy()

            if pred_[0] == 0:
                print('Positive')
            elif pred_[0] == 1:
                print('Negative')
            else:
                print('Neutral')
        print(f'Inference time: {time.time() - start:.2f}')

    elif args.method == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
        model = XLNetClassifier('xlnet-base-cased', pretrained_dim, polarities_dim, dropout)

        encoding = tokenizer.encode_plus(aspect, sentence, return_token_type_ids=True)
        tmp = {}
        tmp['input_ids'] = [pad_and_truncate(encoding['input_ids'], max_seq_len).tolist()]
        tmp['token_type_ids'] = [pad_and_truncate(encoding['token_type_ids'], max_seq_len).tolist()]
        tmp['attention_mask'] = [pad_and_truncate(encoding['attention_mask'], max_seq_len).tolist()]

        test_input_ids = torch.tensor(tmp['input_ids']).to(device)
        test_token_type_ids = torch.tensor(tmp['token_type_ids']).to(device)
        test_attention_mask = torch.tensor(tmp['attention_mask']).to(device)

        model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'{args.method}-{args.dataset}.pth'), map_location=device))
        model.to(device)
        
        start = time.time()
        with torch.no_grad():
            pred = model(input_ids=test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids)
            pred_ = pred.argmax(dim=1).cpu().detach().numpy()

            if pred_[0] == 0:
                print('Positive')
            elif pred_[0] == 1:
                print('Negative')
            else:
                print('Neutral')
        print(f'Inference time: {time.time() - start:.2f}')

    elif args.method == 'xlmr':
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=True)
        model = XLMRobertaClassifier('xlm-roberta-base', pretrained_dim, polarities_dim, dropout)

        encoding = tokenizer.encode_plus(aspect, sentence, return_token_type_ids=True)
        tmp = {}
        tmp['input_ids'] = [pad_and_truncate(encoding['input_ids'], max_seq_len).tolist()]
        tmp['token_type_ids'] = [pad_and_truncate(encoding['token_type_ids'], max_seq_len).tolist()]
        tmp['attention_mask'] = [pad_and_truncate(encoding['attention_mask'], max_seq_len).tolist()]

        test_input_ids = torch.tensor(tmp['input_ids']).to(device)
        test_token_type_ids = torch.tensor(tmp['token_type_ids']).to(device)
        test_attention_mask = torch.tensor(tmp['attention_mask']).to(device)

        model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'{args.method}-{args.dataset}.pth'), map_location=device))
        model.to(device)
        
        start = time.time()
        with torch.no_grad():
            pred = model(input_ids=test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids)
            pred_ = pred.argmax(dim=1).cpu().detach().numpy()

            if pred_[0] == 0:
                print('Positive')
            elif pred_[0] == 1:
                print('Negative')
            else:
                print('Neutral')
        print(f'Inference time: {time.time() - start:.2f}')
    
    else:
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
        start = time.time()
        for i in range(num_model):
            encoding = tokenizers[i].encode_plus(aspect, sentence, return_token_type_ids=True)
            tmp = {}
            tmp['input_ids'] = [pad_and_truncate(encoding['input_ids'], max_seq_len).tolist()]
            tmp['token_type_ids'] = [pad_and_truncate(encoding['token_type_ids'], max_seq_len).tolist()]
            tmp['attention_mask'] = [pad_and_truncate(encoding['attention_mask'], max_seq_len).tolist()]

            test_input_ids = torch.tensor(tmp['input_ids']).to(device)
            test_token_type_ids = torch.tensor(tmp['token_type_ids']).to(device)
            test_attention_mask = torch.tensor(tmp['attention_mask']).to(device)

            models[i].load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'{model_names[i]}-{args.dataset}.pth'), map_location=device))
            models[i].to(device)

            with torch.no_grad():
                pred = models[i](input_ids=test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids)
                all_pred.append(pred.tolist())
        
        all_pred = np.array(all_pred)
        
        if args.method == 'hard-voting':
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
            if meta_pred[0] == 0:
                print('Positive')
            elif meta_pred[0] == 1:
                print('Negative')
            else:
                print('Neutral')
            print(f'Inference time: {time.time() - start:.2f}')
        
        elif args.method == 'soft-voting':
            all_pred = softmax(all_pred, axis=-1)
            all_pred = np.transpose(all_pred, (1, 2, 0))
            all_pred = np.mean(all_pred, axis=-1)
            all_pred = np.argmax(all_pred, axis=-1)
            meta_pred = all_pred
            if meta_pred[0] == 0:
                print('Positive')
            elif meta_pred[0] == 1:
                print('Negative')
            else:
                print('Neutral')
            print(f'Inference time: {time.time() - start:.2f}')
        
        elif args.method == 'weighted_averaging':
            all_pred = softmax(all_pred, axis=-1)
            all_pred = np.transpose(all_pred, (1, 2, 0))

            if args.dataset == 'res14':
                weight = np.array([0.7511, 0.7743, 0.7825, 0.7525])
            else:
                weight = np.array([0.7933, 0.7673, 0.7628, 0.7532])
    
            all_pred = np.average(all_pred, weights=weight, axis=-1)
            all_pred = np.argmax(all_pred, axis=-1)
            meta_pred = all_pred
            if meta_pred[0] == 0:
                print('Positive')
            elif meta_pred[0] == 1:
                print('Negative')
            else:
                print('Neutral')
            print(f'Inference time: {time.time() - start:.2f}')
        
        else:
            all_pred = np.transpose(all_pred, (1, 2, 0))
            all_pred = np.reshape(all_pred, (all_pred.shape[0], -1))
            all_pred = torch.tensor(all_pred, dtype=torch.float).to(device)
    
            meta_model = MLPClassifier(all_pred.shape[-1], hidden_dim, polarities_dim)
            meta_model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'meta-{args.dataset}-blending-mlp.pth'), map_location=device))
            meta_model.to(device)

            with torch.no_grad():
                meta_pred = meta_model(all_pred)
                meta_pred_ = meta_pred.argmax(dim=1).cpu().detach().numpy()
                if meta_pred_[0] == 0:
                    print('Positive')
                elif meta_pred_[0] == 1:
                    print('Negative')
                else:
                    print('Neutral')
            print(f'Inference time: {time.time() - start:.2f}')
