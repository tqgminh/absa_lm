from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
import numpy as np

# tokenizer = RobertaTokenizer.from_pretrained('xlnet-base-cased')
# tokenizer.cls_token_id

def read_data(dataset, path):
    if dataset == 'res14':
        with open(path, 'r') as f:
            data = f.read()

        bsdata = BeautifulSoup(data, "lxml")
        sentence = bsdata.find_all('sentence')
 
        data = []

        for i in range(len(sentence)):
            text = sentence[i].find('text').contents[0]
            aspectCategories = sentence[i].find_all('aspectCategory')

            category_polarity = {}
            for aspectCategory in aspectCategories:
                category = aspectCategory.get('category').replace('/', ' ')
                polarity = aspectCategory.get('polarity')
                if category + polarity in category_polarity:
                    continue
                category_polarity[category + polarity] = 0
                tmp = [text, category, polarity]
                data.append(tmp)
        
        return data
    
    else:
        with open(path, 'r') as f:
            data = f.read()

        bsdata = BeautifulSoup(data, "xml")
        sentence = bsdata.find_all('sentence')
 
        data = []

        for i in range(len(sentence)):
            text = sentence[i].find('text').contents[0]
            opinions = sentence[i].find_all('Opinion')

            category_polarity = {}
            for opinion in opinions:
                category = opinion.get('category').lower().replace('#', ' ').replace('_', ' ')
                polarity = opinion.get('polarity')
                if category + polarity in category_polarity:
                    continue
                category_polarity[category + polarity] = 0 
                tmp = [text, category, polarity]
                data.append(tmp)

        return data

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class ABSADataset(Dataset):
    def __init__(self, dataset, path, tokenizer, model_name, max_seq_len):
        data = read_data(dataset, path)
        all_data = []
        for i in range(len(data)):
            tmp = {}
            if data[i][2] == 'positive':
                tmp['label'] = 0
            elif data[i][2] == 'negative':
                tmp['label'] = 1
            elif data[i][2] == 'neutral':
                tmp['label'] = 2
            else:
                continue
            
            if model_name == 'bert' or model_name == 'roberta' or model_name == 'xlmr':
                encoding = tokenizer.encode_plus(data[i][1], data[i][0], return_token_type_ids=True)
                tmp['input_ids'] = pad_and_truncate(encoding['input_ids'], max_seq_len)
                tmp['token_type_ids'] = pad_and_truncate(encoding['token_type_ids'], max_seq_len)
                tmp['attention_mask'] = pad_and_truncate(encoding['attention_mask'], max_seq_len)
            else:
                encoding = tokenizer(data[i][0] + ' [sep]', data[i][1] + ' [sep] [cls]')
                # encoding = tokenizer(data[i][0], data[i][1])
                tmp['input_ids'] = pad_and_truncate(encoding['input_ids'], max_seq_len)
                tmp['token_type_ids'] = pad_and_truncate(encoding['token_type_ids'], max_seq_len)
                tmp['attention_mask'] = pad_and_truncate(encoding['attention_mask'], max_seq_len)
            
            all_data.append(tmp)
        self.data = all_data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

# dataset = ABSADataset('res16', 'data/res16/ABSA16_Restaurants_Train_SB1_v2.xml', tokenizer, 85)
# print(dataset.__getitem__(0))
class ABSATestDataset(Dataset):
    def __init__(self, dataset, path, tokenizer, model_name, max_seq_len):
        data = read_data(dataset, path)
        all_data = {
            'label': [],
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': []
        }
        
        for i in range(len(data)):
            if data[i][2] == 'positive':
                all_data['label'].append(0)
            elif data[i][2] == 'negative':
                all_data['label'].append(1)
            elif data[i][2] == 'neutral':
                all_data['label'].append(2)
            else:
                continue
            
            if model_name == 'bert' or model_name == 'roberta' or model_name == 'xlmr':
                encoding = tokenizer.encode_plus(data[i][1], data[i][0], return_token_type_ids=True)
                all_data['input_ids'].append(pad_and_truncate(encoding['input_ids'], max_seq_len).tolist())
                all_data['token_type_ids'].append(pad_and_truncate(encoding['token_type_ids'], max_seq_len).tolist())
                all_data['attention_mask'].append(pad_and_truncate(encoding['attention_mask'], max_seq_len).tolist())
            else:
                encoding = tokenizer(data[i][0] + ' [sep]', data[i][1] + ' [sep] [cls]')
                all_data['input_ids'].append(pad_and_truncate(encoding['input_ids'], max_seq_len).tolist())
                all_data['token_type_ids'].append(pad_and_truncate(encoding['token_type_ids'], max_seq_len).tolist())
                all_data['attention_mask'].append(pad_and_truncate(encoding['attention_mask'], max_seq_len).tolist())
            
        self.data = all_data
    
    def __getitem__(self, index):
        return self.data['label'][index], self.data['input_ids'][index], self.data['token_type_ids'][index], self.data['attention_mask'][index]
    
    def __len__(self):
        return len(self.data['label'])


class MetaDataset(Dataset):
    def __init__(self, inp, out):
        all_data = []

        for i in range(inp.shape[0]):
            tmp = {}
            tmp['input'] = torch.tensor(inp[i].tolist())
            tmp['output'] = torch.tensor(out[i])
            all_data.append(tmp)
        
        self.data = all_data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)