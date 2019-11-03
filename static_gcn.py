#!/usr/bin/env python
# coding: utf-8

# In[109]:


import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import BitcoinOTC
import datetime
from dgl.nn.pytorch import GraphConv
import time

import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import matplotlib.pyplot as plt


# ## Hyperparams

# In[99]:


node_dim = 256
n_layers = 2
dropout = 0.2
learning_rate = 0.01
wt_decay = 5e-4
stpsize = 15
n_epochs = 10
out_path = '../btc_static/'
self_loop = True


# In[100]:


num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    device = 'cuda'
else:
    device = 'cpu'


# ## Data preparation

# In[101]:


def removeSelfEdges(edgeList, colFrom, colTo):
    mask = edgeList[:, colFrom] - edgeList[:, colTo] != 0
    edgeList = edgeList[mask]
    return edgeList


# In[102]:


#Trying DGL's implementation of BitcoinOTC class
data = BitcoinOTC()
print("Number of graphs: ")
print(len(data.graphs))
traingr = data.graphs[94]
traingr.number_of_edges()


# In[112]:


#Loading 
graphs = []

data  = np.loadtxt('../soc-sign-bitcoinotc.csv', delimiter=',').astype(np.int64)
data[:, 0:2] = data[:, 0:2] - data[:, 0:2].min()
data = removeSelfEdges(data, 0, 1)
num_nodes = data[:, 0:2].max() - data[:, 0:2].min() + 1
delta = datetime.timedelta(days=14).total_seconds()
time_index = np.around(
    (data[:, 3] - data[:, 3].min())/delta).astype(np.int64)

prevind = 0
for i in range(time_index.max()):
    g = DGLGraph()
    g.add_nodes(num_nodes)
    row_mask = time_index <= i
    edges = data[row_mask][:, 0:2]
    rate = data[row_mask][:, 2]
    diffmask = np.arange(len(edges)) >= prevind
    g.add_edges(edges[:, 0], edges[:, 1])
    g.edata['h'] = torch.FloatTensor(rate.reshape(-1, 1))
    g.edata['diff'] = diffmask
    g.ndata['h'] = torch.zeros(num_nodes, node_dim)
    
    if self_loop == True:
        g.add_edges(g.nodes(), g.nodes())
        
    selfedgemask = np.zeros(g.number_of_edges(), dtype = bool)
    selfedgemask[-g.number_of_nodes():] = True
    g.edata['self_edge'] = selfedgemask
    
    graphs.append(g)
    prevind = len(edges)
    
train_graph = graphs[94]
valid_graphs = graphs[95:109]
test_graphs = graphs[109:]


# ## Model definition

# In[114]:


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.outlayer = nn.Linear(2 * n_hidden, 1, bias = True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, g):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        srcfeatures = torch.stack(list(map(lambda nd: h[nd], g.all_edges()[0])))
        destfeatures = torch.stack(list(map(lambda nd: h[nd], g.all_edges()[1])))
        edgefeatures = torch.cat((srcfeatures, destfeatures), dim = 1)
        outputs = self.outlayer(edgefeatures)
        outputs = 20*torch.sigmoid(outputs) - 1
        return outputs


# ## Training loop

# In[115]:


def evaluate_loss(model, criterion, device, valid_graphs):
    model.eval()
    
    epoch_loss = 0
    num_samples = 0
    
    #validation phase
    with torch.set_grad_enabled(False):
        for val_graph in valid_graphs:
            outputs = model(val_graph.ndata['h'], val_graph)
            labels = val_graph.edata['h']
            outputs = outputs[val_graph.edata['diff']]
            labels = labels[val_graph.edata['diff']]
            num_samples += labels.shape[0] 
            loss = criterion(outputs, labels)
            epoch_loss += (loss.item()* labels.shape[0])
    
    return epoch_loss/num_samples


# In[116]:


#Code for supervised training
def train_model(model, criterion, optimizer, scheduler, device, checkpoint_path, hyperparams, num_epochs=25):
    metrics_dict = {}
    metrics_dict["train"] = {}
    metrics_dict["valid"] = {}
    metrics_dict["train"]["loss"] = {}
    metrics_dict["train"]["loss"]["epochwise"] = []
    metrics_dict["valid"]["loss"] = {}
    metrics_dict["valid"]["loss"]["epochwise"] = []
        
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999999999999999

    for epoch in range(num_epochs):
        print('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('\n')
        
        #train phase
        scheduler.step()
        model.train() 
        optimizer.zero_grad()
        # forward
        # track history if only in train
        forward_start_time  = time.time()
        with torch.set_grad_enabled(True):
            outputs = model(train_graph.ndata['h'], train_graph)
            labels = train_graph.edata['h']
            outputs = outputs[~train_graph.edata['self_edge']]
            labels = labels[~train_graph.edata['self_edge']]
            loss = criterion(outputs, labels)
            epoch_loss = loss.item()
            loss.backward()
            optimizer.step()
        forward_time = time.time() - forward_start_time
        
        print('Train Loss: {:.4f} \n'.format(epoch_loss))
        metrics_dict["train"]["loss"]["epochwise"].append(epoch_loss)
        
        #validation phase
        val_epoch_loss = evaluate_loss(model, criterion, device, valid_graphs)
        print('Validation Loss: {:.4f} \n'.format(val_epoch_loss))
        metrics_dict["valid"]["loss"]["epochwise"].append(val_epoch_loss)
        
        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'full_metrics': metrics_dict,
        'hyperparams': hyperparams
        }, '%s/net_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f} \n'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[ ]:


# create GCN model
model = GCN(node_dim, node_dim, n_layers, F.relu, dropout)
model.to(device)
criterion = nn.MSELoss()
model_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(model_parameters, lr=learning_rate, weight_decay = wt_decay)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=stpsize, gamma=0.1)
hyper_params = {'node_dim' : node_dim,
   'n_layers' : n_layers,
   'dropout' : dropout,
   'wt_decay' : wt_decay }

bst_model = train_model(model, criterion, optimizer, exp_lr_scheduler, device, out_path, hyper_params, n_epochs)


# In[ ]:




