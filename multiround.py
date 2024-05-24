import os
from transformers import BertTokenizer
import torch
from load_data import BertClassifier


bert_name = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = './bert_based_checkpoint'
model = BertClassifier()
model.load_state_dict(torch.load(os.path.join(save_path, 'best.pt')))
model = model.to(device)
model.eval()

real_labels = []
with open('./data/class.txt', 'r') as f:
    for row in f.readlines():
        real_labels.append(row.strip())


if __name__=="__main__":
    while True:
        text = input('请输入用户指令：')
        bert_input = tokenizer(text, 
                            padding='max_length', 
                            max_length = 81, 
                            truncation=True,
                            return_tensors="pt")
        input_ids = bert_input['input_ids'].to(device)
        masks = bert_input['attention_mask'].unsqueeze(1).to(device)
        output = model(input_ids, masks)
        pred = output.argmax(dim=1)
        print(real_labels[pred])
