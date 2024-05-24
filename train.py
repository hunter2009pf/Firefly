'''
Author: feipan3 feipan3@iflytek.com
Date: 2024-05-24 14:09:30
LastEditors: feipan3 feipan3@iflytek.com
LastEditTime: 2024-05-24 14:34:18
FilePath: \Firefly\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import DataLoader
from load_data import BertClassifier, MyDataset, GenerateData


# 训练超参数
epoch = 5
batch_size = 64
learning_rate = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 20240524
save_path = './bert_based_checkpoint'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


if __name__=="__main__":
    setup_seed(random_seed)

    # 定义模型
    model = BertClassifier()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    criterion = criterion.to(device)

    # 构建数据集
    train_dataset = GenerateData(mode='train')
    eval_dataset = GenerateData(mode='eval')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    
    # 训练
    best_eval_acc = 0 # 最佳的评估准确度
    for epoch_num in range(epoch):
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += batch_loss.item()

        # ----------- 验证模型 -----------
        model.eval()
        total_acc_val = 0
        total_loss_val = 0
        
        with torch.no_grad():
            for inputs, labels in eval_loader:
                input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
                masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
                labels = labels.to(device)
                output = model(input_ids, masks)

                batch_loss = criterion(output, labels)
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc
                total_loss_val += batch_loss.item()
            
            print(f'''Epochs: {epoch_num + 1} 
            | Train Loss: {total_loss_train / len(train_dataset): .3f} 
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
            | Val Loss: {total_loss_val / len(eval_dataset): .3f} 
            | Val Accuracy: {total_acc_val / len(eval_dataset): .3f}''')
            
            # 保存最优的模型
            if total_acc_val / len(eval_dataset) > best_eval_acc:
                best_eval_acc = total_acc_val / len(eval_dataset)
                save_model('best.pt')
            
        model.train()

    # 保存最后的模型
    save_model('last.pt')
