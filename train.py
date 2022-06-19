import logging
import argparse
import sys
import time
import os

from sklearn import metrics
from time import strftime, localtime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, RMSprop, Adagrad
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer, XLMRobertaTokenizer

from data_utils import ABSADataset
from models import BertClassifier, RobertaClassifier, XLNetClassifier, XLMRobertaClassifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', default='bert', type=str, help='bert, roberta or xlnet')
    parser.add_argument('--dataset', default='res14', type=str, help='res14 or res16')
    parser.add_argument('--optimizer', default='adam', type=str, help='adam or adagrad or rmsprop or sgd')
    
    parser.add_argument('--lr', default=1e-5, type=float, help='try 1e-5, 2e-5 or 5e-5')
    parser.add_argument('--dropout', default=0.1, type=float)
    
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=64, type=int, help='16, 32, 64 or 128')
    parser.add_argument('--log_step', default=10, type=int)
    
    parser.add_argument('--pretrained_dim', default=768, type=int, help='768 for base model')
    parser.add_argument('--pretrained_name', default='bert-base-uncased', type=str, help='bert: bert-base-uncased; roberta: roberta-base')
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int, help='3 for positive, negative and neutral')
    
    parser.add_argument('--device', default=None, type=str, help='cpu or gpu')

    args = parser.parse_args()

    log_folder = 'log'
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if not os.path.exists(os.path.join(log_folder, args.dataset)):
        os.mkdir(os.path.join(log_folder, args.dataset))
    log_file = f'{args.model_name}-{args.dataset}-{strftime("%d%m%y-%H%M", localtime(time.time()+7*3600))}.log'
    logger.addHandler(logging.FileHandler(os.path.join(log_folder, args.dataset, log_file)))

    logger.info(f'- Model name: {args.model_name}')
    logger.info(f'- Dataset: {args.dataset}')
    logger.info(f'- Optimizer: {args.optimizer}')
    logger.info(f'- Learning rate: {args.lr}')
    logger.info(f'- Dropout: {args.dropout}')
    logger.info(f'- Num epoch: {args.num_epoch}')
    logger.info(f'- Batch size: {args.batch_size}')
    logger.info(f'- Pretrained name: {args.pretrained_name}')
    logger.info(f'- Pretrained dim: {args.pretrained_dim}')
    logger.info(f'- Max seq len: {args.max_seq_len}')
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
    
    if args.model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_name, do_lower_case=True)
        model = RobertaClassifier(args.pretrained_name, args.pretrained_dim, args.polarities_dim, args.dropout)
    elif args.model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_name, do_lower_case=True)
        model = BertClassifier(args.pretrained_name, args.pretrained_dim, args.polarities_dim, args.dropout)
    elif args.model_name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained(args.pretrained_name, do_lower_case=True)
        model = XLNetClassifier(args.pretrained_name, args.pretrained_dim, args.polarities_dim, args.dropout)
    else:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_name, do_lower_case=True)
        model = XLMRobertaClassifier(args.pretrained_name, args.pretrained_dim, args.polarities_dim, args.dropout)


    
    train_data = ABSADataset(args.dataset, dataset_files[args.dataset]['train'], tokenizer, args.model_name, args.max_seq_len)
    train_data, dev_data = random_split(train_data, [train_data.__len__()-300, 300], generator=torch.Generator().manual_seed(42))
    logger.info(f'Train size: {train_data.__len__()}')
    logger.info(f'Dev size: {dev_data.__len__()}')
    
    test_data = ABSADataset(args.dataset, dataset_files[args.dataset]['test'], tokenizer, args.model_name, args.max_seq_len)
    logger.info(f'Test size: {test_data.__len__()}\n')

    
    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    dev_data_loader = DataLoader(dataset=dev_data, batch_size=args.batch_size, shuffle=False)
    test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    use_cuda = torch.cuda.is_available()
    if args.device == "cpu":
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adagrad':
        optimizer = Adagrad(model.parameters(), lr=args.lr)
    else:
        optimizer = RMSprop(model.parameters(), lr=args.lr)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    

    weight_folder = 'weight'
    if not os.path.exists(weight_folder):
        os.mkdir(weight_folder)
    if not os.path.exists(os.path.join(weight_folder, args.dataset)):
        os.mkdir(os.path.join(weight_folder, args.dataset))

    max_f1_dev = 0

    for epoch in range(args.num_epoch):

        logger.info(f'Epoch {epoch+1}|{args.num_epoch}:')
        start_time = time.time()
        
        total_acc_train = 0
        total_loss_train = 0

        for (batch, data) in enumerate(train_data_loader):
            
            train_label = data['label'].to(device)
            
            train_input_ids = data['input_ids'].to(device)
            train_token_type_ids = data['token_type_ids'].to(device)
            train_attention_mask = data['attention_mask'].to(device)

            output = model(input_ids=train_input_ids, attention_mask=train_attention_mask, token_type_ids=train_token_type_ids)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = metrics.accuracy_score(train_label.cpu().detach().numpy(), output.argmax(dim=1).cpu().detach().numpy())
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if batch % args.log_step == 0 or batch == len(train_data_loader)-1:
                logger.info(f'Batch {batch+1}|{len(train_data_loader)}: loss {batch_loss:.4f} accuracy {acc:.4f}')
        
        logger.info(f'Loss {total_loss_train/len(train_data_loader):.4f} accuracy {total_acc_train/len(train_data_loader):.4f}')
        logger.info(f'Time: {time.time() - start_time:.2f}')

        total_acc_dev = 0
        total_loss_dev = 0
        total_f1_dev = 0
        
        with torch.no_grad():
            for (batch, data) in enumerate(dev_data_loader):
                
                dev_label = data['label'].to(device)

                dev_input_ids = data['input_ids'].to(device)
                dev_token_type_ids = data['token_type_ids'].to(device)
                dev_attention_mask = data['attention_mask'].to(device)

                output = model(input_ids=dev_input_ids, attention_mask=dev_attention_mask, token_type_ids=dev_token_type_ids)
                
                batch_loss = criterion(output, dev_label)
                total_loss_dev += batch_loss.item()

                acc = metrics.accuracy_score(dev_label.cpu().detach().numpy(), output.argmax(dim=1).cpu().detach().numpy())
                total_acc_dev += acc

                f1 = metrics.f1_score(dev_label.cpu().detach().numpy(), output.argmax(dim=1).cpu().detach().numpy(), average='macro')
                total_f1_dev += f1
        
        logger.info(f'Dev: loss {total_loss_dev/len(dev_data_loader):.4f} accuracy {total_acc_dev/len(dev_data_loader):.4f} f1 {total_f1_dev/len(dev_data_loader):.4f}')
        
        if max_f1_dev < total_f1_dev/len(dev_data_loader):
            max_f1_dev = total_f1_dev/len(dev_data_loader)
            torch.save(model.state_dict(), os.path.join(weight_folder, args.dataset, f'{args.model_name}-{args.dataset}.pth'))
            logger.info(f'Save model weight !')
        
        logger.info('')
    
    model.load_state_dict(torch.load(os.path.join(weight_folder, args.dataset, f'{args.model_name}-{args.dataset}.pth'), map_location=device))
    logger.info('Restore best model !')
    test_pred = []
    test_output = []

    model.eval()
    for (batch, data) in enumerate(test_data_loader):
                
        test_label = data['label'].to(device)

        test_input_ids = data['input_ids'].to(device)
        test_token_type_ids = data['token_type_ids'].to(device)
        test_attention_mask = data['attention_mask'].to(device)

        pred = model(input_ids=test_input_ids, attention_mask=test_attention_mask, token_type_ids=test_token_type_ids)
        pred = pred.argmax(dim=1).cpu().detach().numpy()
        test_pred = [*test_pred, *pred]

        output = test_label.cpu().detach().numpy()
        test_output = [*test_output, *output]
        
    logger.info(f'Test: accuracy {metrics.accuracy_score(test_output, test_pred):.4f} f1 {metrics.f1_score(test_output, test_pred, average="macro"):.4f} \n')
