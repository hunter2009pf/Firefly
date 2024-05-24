import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel

bert_name = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)

class MyDataset(Dataset):
    def __init__(self, df):
        # tokenizer分词后可以被自动汇聚
        self.texts = [tokenizer(text, 
                                padding='max_length',  # 填充到最大长度
                                max_length = 81, 	# 经过数据分析，最大长度为35
                                truncation=True,
                                return_tensors="pt") 
                      for text in df['text']]
        # Dataset会自动返回Tensor
        self.labels = [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
    
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def GenerateData(mode):
    train_data_path = './data/train.txt'
    test_data_path = './data/test.txt'
    eval_data_path = './data/eval.txt'

    train_df = pd.read_csv(train_data_path, sep='\t', header=None)
    test_df = pd.read_csv(test_data_path, sep='\t', header=None)
    eval_df = pd.read_csv(eval_data_path, sep='\t', header=None)

    new_columns = ['text', 'label']  
    train_df = train_df.rename(columns=dict(zip(train_df.columns, new_columns)))
    test_df = test_df.rename(columns=dict(zip(test_df.columns, new_columns)))
    eval_df = eval_df.rename(columns=dict(zip(eval_df.columns, new_columns)))

    train_dataset = MyDataset(train_df)
    test_dataset = MyDataset(test_df)
    eval_dataset = MyDataset(eval_df)
    
    if mode == 'train':
        return train_dataset
    elif mode == 'test':
        return test_dataset
    elif mode == 'eval':
        return eval_dataset
