# python3.5 run_sc.py <test_file_path> <model_file_path> <output_file_path>

import os
import math
import sys
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import gzip
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

def test_model(test_text_file, model_file, out_file):
    # write your code here. You can add functions as well.
		# use torch library to load model_file
    embeds = torch.load("embedded.pt")
    word2idx = torch.load("word2idx.pt")
    class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            #embedded = nn.Embedding(len(word2idx), emb_dim, _weight=embeddings)
            self.conv1 = nn.Conv1d(1,16,3*300,padding=2*300,stride=300)
            self.conv2 = nn.Conv1d(1,16, 4 * 300, padding=2 * 300, stride=300)
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

    #net = Net()

    class SentimentDataset(Dataset):
        def __init__(self, trfile,transform=None):
            self.x_test = [[line.rstrip('\n')] for line in open(trfile)]
            #self.y_train = [[line.rstrip('\n')] for line in open(tefile)]
            self.transform = transform
            print(len(self.x_test));

        def __len__(self):
            return len(self.x_test)

        def __getitem__(self, idx):
            txt = self.x_test[idx][0];
            final_txt=[];
            for i in txt.lower().split():
                if i not in word2idx:
                    continue;
                else:
                    lookup_tensor = embeds(torch.tensor([word2idx[i]], dtype=torch.long))
                final_txt.append(lookup_tensor);
            label = -1;
            sample = {'text': torch.cat(final_txt,dim=1), 'label': label};
            return sample

    sentiment_dataset = SentimentDataset(trfile=test_text_file);

    testloader = DataLoader(sentiment_dataset, batch_size=1, shuffle=False)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load(model_file));
    model.to(device)
    ans=[];
    print("Starting forward propagation")
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['text'],data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            #print("Doing forward prop.")
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            ans.append((predicted.item())+1);
    #print(len(ans))
    #print(ans)
    with open(out_file, 'w') as f:
        for k in ans:
            f.write("%i\n" % k)

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_text_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    test_model(test_text_file, model_file, out_file)
