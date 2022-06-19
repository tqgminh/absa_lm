import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, RobertaModel, XLNetModel, XLMRobertaModel

class BertClassifier(nn.Module):
    def __init__(self, pretrained_name, pretrained_dim, polarities_dim, dropout):
        super(BertClassifier, self).__init__()
        self.model = BertModel.from_pretrained(pretrained_name)
        self.drop = nn.Dropout(dropout)
        self.final_layer = nn.Linear(pretrained_dim, polarities_dim)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        out = self.drop(pooled_output)
        out = self.final_layer(out)

        return out

class RobertaClassifier(nn.Module):
    def __init__(self, pretrained_name, pretrained_dim, polarities_dim, dropout):
        super(RobertaClassifier, self).__init__()
        self.model = RobertaModel.from_pretrained(pretrained_name)
        self.drop = nn.Dropout(dropout)
        self.final_layer = nn.Linear(pretrained_dim, polarities_dim)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        out = self.drop(pooled_output)
        out = self.final_layer(out)

        return out

class XLNetClassifier(nn.Module):
    def __init__(self, pretrained_name, pretrained_dim, polarities_dim, dropout):
        super(XLNetClassifier, self).__init__()
        self.model = XLNetModel.from_pretrained(pretrained_name)
        self.drop = nn.Dropout(dropout)
        self.final_layer = nn.Linear(pretrained_dim, polarities_dim)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        last_hidden_states, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        pooled_output = torch.squeeze(last_hidden_states[:, -1, :], dim=1)
        # pooled_output = torch.mean(last_hidden_states, dim=1)
        out = self.drop(pooled_output)
        out = self.final_layer(out)
        
        return out
    
class XLMRobertaClassifier(nn.Module):
    def __init__(self, pretrained_name, pretrained_dim, polarities_dim, dropout):
        super(XLMRobertaClassifier, self).__init__()
        self.model = XLMRobertaModel.from_pretrained(pretrained_name)
        self.drop = nn.Dropout(dropout)
        self.final_layer = nn.Linear(pretrained_dim, polarities_dim)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        out = self.drop(pooled_output)
        out = self.final_layer(out)
        
        return out
    

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.last_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inp):
        inp = self.layernorm(inp)
        hidden = self.first_layer(inp)
        hidden = F.relu(hidden)
        out = self.last_layer(hidden)

        return out

class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LinearClassifier, self).__init__()
        self.weight = nn.Linear(input_dim, 1)
    
    def forward(self, inp):
        out = self.weight(inp)
        out = torch.squeeze(out)

        return out
