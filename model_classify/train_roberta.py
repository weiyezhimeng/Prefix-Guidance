from load_data import load_data
from train import train 
import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn

epoch=10
batch=64
file_train="../I'm_sorry/deny.csv"
device ='cuda:0'
tokenizer_path = model_path = "roberta-base"
lr=1e-5
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, padding_side='right')
model_bert = RobertaModel.from_pretrained(model_path)

class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = model_bert
        self.classifier = nn.Linear(768, num_labels)

        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        for name, param in self.classifier.named_parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[1]
        logits = self.classifier(last_hidden_state_cls)
        return logits

num_labels = 2
model = BertClassifier(num_labels).to(device)

loader_train = load_data(file_train, batch)

train(tokenizer, model, device, lr, epoch, loader_train)