# python3.5 build_sc.py <pretrained_vectors_gzipped_file_absolute_path> <train_text_path> <train_label_path> <model_file_path>

import os
import math
import sys
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import gzip
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

def train_model(embeddings_file, train_text_file, train_label_file, model_file):
    # Create word2idx
    with open(train_text_file)as f:
        vocab = set(f.read().lower().split());
    word2idx = {};
    word2idx = dict([ (elem, ind) for ind,elem in enumerate(vocab) ])
    f.close();

    # Create embedding
    with gzip.open("/home/course/cs5246/vectors.txt.gz", "rt", encoding="utf-8") as f:
        first_line = f.readline().strip()
        first_line_tokens = first_line.split(' ')
        emb_dim = int(first_line_tokens[1])
        print(emb_dim)
        embeddings = torch.rand(len(word2idx), emb_dim) * 0.5 - 0.25
        for line in f:
            line = line.strip()
            first_space_pos = line.find(' ', 1)
            word = line[:first_space_pos]
            if word not in word2idx:
                continue
            idx = word2idx[word]
            emb_str = line[first_space_pos + 1:].strip()
            emb = [float(t) for t in emb_str.split(' ')]
            embeddings[idx] = torch.tensor(emb)
        f.close()
    embeds = nn.Embedding(len(word2idx), emb_dim, _weight=embeddings)
    #print(embeds)
    print('Created embedding...')
    #lookup_tensor = torch.tensor([word2idx["wrann"]], dtype=torch.long)
    #hello_embed = embeds(lookup_tensor)
    #print("wrann: ",hello_embed)

    # Create NN Architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            #embedded = nn.Embedding(len(word2idx), emb_dim, _weight=embeddings)
            self.conv1 = nn.Conv1d(1,16,3*300,padding=2*300,stride=300)
            self.conv2 = nn.Conv1d(1, 16, 4 * 300, padding=2 * 300, stride=300)
            self.fc = nn.Linear(32, 2)
            #self.size = n;
        def forward(self,x):
            x1 = F.relu(self.conv1(x))
            x1 = F.max_pool1d(x1,x1.size(2))
            x2 = F.relu(self.conv2(x))
            x2 = F.max_pool1d(x2,x2.size(2))
            x3 = torch.cat((x1,x2), dim=1)
            x4 = x3.view(-1,32)
            x5 = self.fc(x4);
            #print("x5 size: ",x5.size())
            return x5

    net = Net()

    # GPU Convert
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    print(net)

    class SentimentDataset(Dataset):
        def __init__(self, trfile, tefile, transform=None):
            self.x_train = [[line.rstrip('\n')] for line in open(trfile)]
            self.y_train = [[line.rstrip('\n')] for line in open(tefile)]
            self.transform = transform
            print(len(self.x_train));

        def __len__(self):
            return len(self.x_train)

        def __getitem__(self, idx):
            txt = self.x_train[idx][0];
            final_txt=[];
            for i in txt.lower().split():
                lookup_tensor = embeds(torch.tensor([word2idx[i]], dtype=torch.long))
                final_txt.append(lookup_tensor);
            label = int(self.y_train[idx][0])-1;
            sample = {'text': torch.cat(final_txt,dim=1), 'label': label};
            return sample

    sentiment_dataset = SentimentDataset(trfile=train_text_file,tefile=train_label_file);

    '''for i in range(len(sentiment_dataset)):
        sample = sentiment_dataset[i];
        print(i, sample['text'].shape)

        if i==3:
            break;'''
    '''
   for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['text'],sample_batched['label'])
        if(i_batch==2):
            break;
    '''

    # Data Loader
    trainloader = DataLoader(sentiment_dataset, batch_size=1, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Start Training
    for epoch in range(2):
        running_loss = 0.0
        for i, sample_batched in enumerate(trainloader,0):
            inputs,labels = sample_batched['text'],sample_batched['label']
            inputs,labels = inputs.to(device),labels.to(device)
            #print(inputs);
            #print(labels.shape)
            optimizer.zero_grad()

            outputs = net(inputs)
           # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training....')

    # Save model,word2idx,embedding
    embed_file = "embedded.pt"
    word2idx_file = "word2idx.pt"
    #torch.save(embeds,embeddings_file)
    torch.save(net.state_dict(),model_file)
    torch.save(embeds, embed_file)
    print("Saving word2idx")
    torch.save(word2idx, word2idx_file)
    print("Saved word2idx")

if __name__ == "__main__":
    embeddings_file = sys.argv[1]
    train_text_file = sys.argv[2]
    train_label_file = sys.argv[3]
    model_file = sys.argv[4]
    train_model(embeddings_file, train_text_file, train_label_file, model_file)