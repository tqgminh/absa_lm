import logging
import argparse
import sys
import time
import os

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from time import strftime, localtime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, RMSprop, Adagrad
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer, XLMRobertaTokenizer

from data_utils import ABSADataset, MetaDataset, ABSATestDataset
from models import BertClassifier, RobertaClassifier, XLNetClassifier, XLMRobertaClassifier, MLPClassifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='res14', type=str, help='res14 or res16')
    parser.add_argument('--optimizer', default='adam', type=str, help='adam or adagrad or rmsprop or sgd')
    
    parser.add_argument('--lr', default=0.001, type=float, help='try 1e-5, 2e-5 or 5e-5')
    
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='16, 32, 64 or 128')
    parser.add_argument('--log_step', default=10, type=int)
    
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int, help='3 for positive, negative and neutral')
    
    parser.add_argument('--device', default=None, type=str, help='cpu or gpu')

    args = parser.parse_args()

    log_folder = 'log'
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if not os.path.exists(os.path.join(log_folder, args.dataset)):
        os.mkdir(os.path.join(log_folder, args.dataset))
    log_file = f'meta-{args.dataset}-mlp-blending-{strftime("%d%m%y-%H%M", localtime(time.time()+7*3600))}.log'
    logger.addHandler(logging.FileHandler(os.path.join(log_folder, args.dataset, log_file)))

    logger.info(f'- Dataset: {args.dataset}')
    logger.info(f'- Optimizer: {args.optimizer}')
    logger.info(f'- Learning rate: {args.lr}')
    logger.info(f'- Num epoch: {args.num_epoch}')
    logger.info(f'- Batch size: {args.batch_size}')
    logger.info(f'- Hidden dim: {args.hidden_dim}')
    logger.info(f'- Polarities dim: {args.polarities_dim}')
    logger.info(f'- Device: {args.device} \n')
    
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

    use_cuda = torch.cuda.is_available()
    if args.device == "cpu":
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    weight_folder = 'weight'
    if not os.path.exists(weight_folder):
        os.mkdir(weight_folder)
    if not os.path.exists(os.path.join(weight_folder, args.dataset)):
        os.mkdir(os.path.join(weight_folder, args.dataset))

    max_f1_dev = 0
    model_names = ['roberta', 'bert', 'xlnet', 'xlmr']
    pretrained_dim = 768
    dropout = 0.1
    max_seq_len = 50

    meta_input = []
    meta_output = []

    for model_name in model_names:
        
        if model_name == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
            model = RobertaClassifier('roberta-base', pretrained_dim, args.polarities_dim, dropout)
        elif model_name == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = BertClassifier('bert-base-uncased', pretrained_dim, args.polarities_dim, dropout)
        elif model_name == 'xlnet':
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
            model = XLNetClassifier('xlnet-base-cased', pretrained_dim, args.polarities_dim, dropout)
        else:
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=True)
            model = XLMRobertaClassifier('xlm-roberta-base', pretrained_dim, args.polarities_dim, dropout)


        train_data = ABSADataset(args.dataset, dataset_files[args.dataset]['train'], tokenizer, model_name, max_seq_len)
        train_data, dev_data = random_split(train_data, [train_data.__len__()-300, 300], generator=torch.Generator().manual_seed(42))
        test_data = ABSADataset(args.dataset, dataset_files[args.dataset]['test'], tokenizer, model_name, max_seq_len)

        train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dataset=dev_data, batch_size=args.batch_size, shuffle=False)
        test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

        model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'{model_name}-{args.dataset}.pth'), map_location=device))
        model.to(device)
        logger.info(f'Restore {model_name} model !')
            
        
        preds = []
        outs = []

        total_acc_dev = 0
        total_f1_dev = 0

        model.eval()
        for (batch, data) in enumerate(dev_data_loader):
                
            dev_label = data['label'].to(device)

            dev_input_ids = data['input_ids'].to(device)
            dev_token_type_ids = data['token_type_ids'].to(device)
            dev_attention_mask = data['attention_mask'].to(device)

            pred = model(input_ids=dev_input_ids, attention_mask=dev_attention_mask, token_type_ids=dev_token_type_ids)

            acc = metrics.accuracy_score(dev_label.cpu().detach().numpy(), pred.argmax(dim=1).cpu().detach().numpy())
            total_acc_dev += acc

            f1 = metrics.f1_score(dev_label.cpu().detach().numpy(), pred.argmax(dim=1).cpu().detach().numpy(), average='macro')
            total_f1_dev += f1  

            pred = pred.cpu().detach().numpy().tolist()
            preds = [*preds, *pred]

            out = dev_label.cpu().detach().numpy().tolist()
            outs = [*outs, *out]
            
        logger.info(f'Dev: accuracy {total_acc_dev/len(dev_data_loader):.4f} f1 {total_f1_dev/len(dev_data_loader):.4f}\n')
            
        meta_input.append(preds)
        meta_output.append(outs)

    
    meta_input = np.array(meta_input)
    meta_input = np.transpose(meta_input, (1, 2, 0))
    meta_input = np.reshape(meta_input, (meta_input.shape[0], -1))
    input_dim = meta_input.shape[1]
    
    meta_output = np.array(meta_output[0])
    
    
    train_meta_data = MetaDataset(meta_input, meta_output)
    train_meta_data, dev_meta_data = random_split(train_meta_data, [train_meta_data.__len__()-100, 100])

    train_meta_data_loader = DataLoader(dataset=train_meta_data, batch_size=args.batch_size, shuffle=True)
    dev_meta_data_loader = DataLoader(dataset=dev_meta_data, batch_size=args.batch_size, shuffle=True)

    meta_model = MLPClassifier(input_dim, args.hidden_dim, args.polarities_dim)
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(meta_model.parameters(), lr=args.lr)
    if args.optimizer == 'sgd':
        optimizer = SGD(meta_model.parameters(), lr=args.lr)
    elif args.optimizer == 'adagrad':
        optimizer = Adagrad(meta_model.parameters(), lr=args.lr)
    else:
        optimizer = RMSprop(meta_model.parameters(), lr=args.lr)

    if use_cuda:
        meta_model = meta_model.cuda()
        criterion = criterion.cuda()
    
    max_f1_dev = 0

    logger.info(f'Start training meta model!\n')
    for epoch in range(args.num_epoch):

        logger.info(f'Epoch {epoch+1}|{args.num_epoch}:')
        start_time = time.time()
        
        total_acc_train = 0
        total_loss_train = 0
        total_f1_train = 0

        for (batch, data) in enumerate(train_meta_data_loader):
            
            train_meta_label = data['output'].to(device)
            
            train_meta_input = data['input'].to(device)

            output = meta_model(train_meta_input)

            batch_loss = criterion(output, train_meta_label)
            total_loss_train += batch_loss.item()

            acc = metrics.accuracy_score(train_meta_label.cpu().detach().numpy(), output.argmax(dim=1).cpu().detach().numpy())
            total_acc_train += acc

            f1 = metrics.f1_score(train_meta_label.cpu().detach().numpy(), output.argmax(dim=1).cpu().detach().numpy(), average='macro')
            total_f1_train += f1

            meta_model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if batch % args.log_step == 0 or batch == len(train_data_loader)-1:
                logger.info(f'Batch {batch+1}|{len(train_meta_data_loader)}: loss {batch_loss:.4f} accuracy {acc:.4f}')
        
        logger.info(f'Loss {total_loss_train/len(train_meta_data_loader):.4f} accuracy {total_acc_train/len(train_meta_data_loader):.4f} f1 {total_f1_train/len(train_meta_data_loader):.4f}')
        logger.info(f'Time: {time.time() - start_time:.2f}')

        
        total_acc_dev = 0
        total_loss_dev = 0
        total_f1_dev = 0
        
        with torch.no_grad():
            for (batch, data) in enumerate(dev_meta_data_loader):
                
                dev_meta_label = data['output'].to(device)
            
                dev_meta_input = data['input'].to(device)

                output = meta_model(dev_meta_input)
                
                batch_loss = criterion(output, dev_meta_label)
                total_loss_dev += batch_loss.item()

                acc = metrics.accuracy_score(dev_meta_label.cpu().detach().numpy(), output.argmax(dim=1).cpu().detach().numpy())
                total_acc_dev += acc

                f1 = metrics.f1_score(dev_meta_label.cpu().detach().numpy(), output.argmax(dim=1).cpu().detach().numpy(), average='macro')
                total_f1_dev += f1
        
        logger.info(f'Dev: loss {total_loss_dev/len(dev_meta_data_loader):.4f} accuracy {total_acc_dev/len(dev_meta_data_loader):.4f} f1 {total_f1_dev/len(dev_meta_data_loader):.4f}')
        
        if max_f1_dev < total_f1_dev/len(dev_meta_data_loader):
            max_f1_dev = total_f1_dev/len(dev_meta_data_loader)
            torch.save(meta_model.state_dict(), os.path.join(weight_folder, args.dataset, f'meta-{args.dataset}-blending-mlp.pth'))
            logger.info(f'Save model weight !')
        
        logger.info('')
    
    meta_model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'meta-{args.dataset}-blending-mlp.pth'), map_location=device))
    logger.info('Restore best meta model !')
    
    all_pred = []

    for model_name in model_names:

        if model_name == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
            model = RobertaClassifier('roberta-base', pretrained_dim, args.polarities_dim, dropout)
        elif model_name == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = BertClassifier('bert-base-uncased', pretrained_dim, args.polarities_dim, dropout)
        elif model_name == 'xlnet':
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
            model = XLNetClassifier('xlnet-base-cased', pretrained_dim, args.polarities_dim, dropout)
        else:
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=True)
            model = XLMRobertaClassifier('xlm-roberta-base', pretrained_dim, args.polarities_dim, dropout)
        
        test_data = ABSATestDataset(args.dataset, dataset_files[args.dataset]['test'], tokenizer, model_name, max_seq_len)

        test_label = torch.tensor(test_data.data['label']).to(device)

        test_input_ids = torch.tensor(test_data.data['input_ids']).to(device)
        test_token_type_ids = torch.tensor(test_data.data['token_type_ids']).to(device)
        test_attention_mask = torch.tensor(test_data.data['attention_mask']).to(device)

        model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'{model_name}-{args.dataset}.pth')))
        model.to(device)
        logger.info(f'Restore {model_name} model !')
        
        with torch.no_grad():
            pred = model(input_ids=test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids)
            pred_ = pred.argmax(dim=1).cpu().detach().numpy()
            
            output = test_label.cpu().detach().numpy()
            logger.info(f'Test: accuracy {metrics.accuracy_score(output, pred_):.4f} f1 {metrics.f1_score(output, pred_, average="macro"):.4f} \n')
            
            all_pred.append(pred.tolist())
    
    meta_model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'meta-{args.dataset}-blending-mlp.pth'), map_location=device))
    logger.info('Restore meta model !\n')

    all_pred = np.array(all_pred)
    all_pred = np.transpose(all_pred, (1, 2, 0))
    all_pred = np.reshape(all_pred, (all_pred.shape[0], -1))
    all_pred = torch.tensor(all_pred, dtype=torch.float).to(device)

    with torch.no_grad():
        meta_pred = meta_model(all_pred)
        meta_pred_ = meta_pred.argmax(dim=1).cpu().detach().numpy()

        logger.info(f'Meta: accuracy {metrics.accuracy_score(output, meta_pred_):.4f} f1 {metrics.f1_score(output, meta_pred_, average="macro"):.4f} \n')
   