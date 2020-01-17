"""
Model Interface
"""
import copy
import importlib
import torch
import numpy as np
import scipy.sparse as sp

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import dgl
from dgl import DGLGraph
from utils.utils import compute_node_degrees
from utils.constants import *

from utils.utils import preprocess_adj

from models.layers.edgnn import edGNNLayer
from models.layers.rgcn import RGCNLayer

ACTIVATIONS = {
    'relu': F.relu
}

class Model(nn.Module):

    def __init__(self, g, config_params, n_classes=None, n_rels=None, n_entities=None, is_cuda=False, seq_dim=None, batch_size=1):
        """
        Instantiate a graph neural network.

        Args:
            g (DGLGraph): a preprocessed DGLGraph
            config_json (str): path to a configuration JSON file. It must contain the following fields: 
                               "layer_type", and "layer_params". 
                               The "layer_params" should be a (nested) dictionary containing at least the fields 
                               "n_units" and "activation". "layer_params" should contain other fields that corresponds
                               to keyword arguments of the concrete layers (refer to the layers implementation).
                               The name of these additional fields should be the same as the keyword args names.
                               The parameters in "layer_params" should either be lists with the same number of elements,
                               or single values. If single values are specified, then a "n_hidden_layers" (integer) 
                               field is expected.
                               The fields "n_input" and "n_classes" are required if not specified 
        """
        super(Model, self).__init__()

        self.is_cuda = is_cuda
        self.config_params = config_params
        self.n_rels = n_rels
        self.n_classes = n_classes
        self.n_entities = n_entities
        self.g = g
        self.seq_dim = seq_dim # number of nodes in a sequence
        self.batch_size = batch_size

        layer_type = config_params["layer_type"]

        self.build_model()

    def build_model(self):
        """
        Build NN
        """
        self.edgnn_layers = nn.ModuleList()
        layer_params = self.config_params['layer_params']

        #######################
        # Edge embeddings
        #######################
        # if 'edge_dim' in self.config_params:
        #     edge_dim = self.config_params['edge_dim']
        #     # self.embed_edges = nn.Embedding(self.n_rels, edge_dim)
        # elif 'edge_one_hot' in self.config_params and self.config_params['edge_one_hot'] is True:
        #     edge_dim = self.n_rels
        #     self.embed_edges = torch.eye(edge_dim, edge_dim)
        #     if self.is_cuda:
        #         self.embed_edges = self.embed_edges.cuda()
        # else:
        #     edge_dim = self.n_rels
        #     self.embed_edges = None

        #######################
        # Node embeddings
        #######################
        # if 'node_dim' in self.config_params:
        #     node_dim = self.config_params['node_dim']
        #     # self.embed_nodes = nn.Embedding(self.n_entities, node_dim)
        # elif 'node_one_hot' in self.config_params and self.config_params['node_one_hot'] is True:
        #     node_dim = self.n_entities
        #     self.embed_nodes = torch.eye(self.n_entities, self.n_entities)
        #     if self.is_cuda:
        #         self.embed_nodes = self.embed_nodes.cuda()
        # else:
        #     node_dim = self.n_entities
        #     # print('node_dim', node_dim)
        #     self.embed_nodes = None
        

        # basic tests
        assert (self.n_classes is not None)

        ############################
        # Build and append layers
        ############################
        print('\n*** Building model ***')
        # self.node_dim = node_dim
        # self.edge_dim = edge_dim
        self.node_dim = self.g[0].ndata[GNN_NODE_TYPES_KEY].shape[1] + self.g[0].ndata[GNN_NODE_LABELS_KEY].shape[1]
        self.edge_dim = self.g[0].edata[GNN_EDGE_TYPES_KEY].shape[1] + self.g[0].edata[GNN_EDGE_LABELS_KEY].shape[1]

        """ edGNN layers """
        n_edGNN_layers = len(layer_params['n_units'])
        for i in range(n_edGNN_layers):
            if i == 0:  # take input from GAT layer
                print('* Building new GNN layer with args:', self.node_dim, self.edge_dim,
                      layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]])

                edGNN = edGNNLayer(self.g, self.node_dim, self.edge_dim,
                                   layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]], is_cuda=self.is_cuda)
            else:
                print('* Building new GNN layer with args:', layer_params['n_units'][i-1], self.edge_dim,
                      layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]])

                edGNN = edGNNLayer(self.g, layer_params['n_units'][i-1], self.edge_dim,
                                   layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]], is_cuda=self.is_cuda)

            # self.add_module('edGNN_{}'.format(i), edGNN)
            self.edgnn_layers.append(edGNN)
        
        """ Classification layer """
        print('* Building fc layer with args:', layer_params['n_units'][-1], self.n_classes)
        self.fc = nn.Linear(layer_params['n_units'][-1], self.n_classes)

        print('*** Model successfully built ***\n')


    def forward(self, g):
        # print(g)

        if g is not None:
            g.set_n_initializer(dgl.init.zero_initializer)
            g.set_e_initializer(dgl.init.zero_initializer)
            self.g = g

        ############################
        # 1. Build node features
        ############################
        # print('\t GNN_NODE_TYPES_KEY.shape', self.g.ndata[GNN_NODE_TYPES_KEY].size())
        # print('\t GNN_NODE_LABELS_KEY.shape', self.g.ndata[GNN_NODE_LABELS_KEY].size())
        # node_features = self.g.ndata[GNN_NODE_LABELS_KEY]
        node_features = torch.cat((self.g.ndata[GNN_NODE_TYPES_KEY], self.g.ndata[GNN_NODE_LABELS_KEY]), dim=1)

        # print('\tnode_features', node_features)
        # node_features = node_features.view(node_features.size()[0], -1)
        # self.node_dim = node_features.size()[1]
        if self.is_cuda:
            node_features = node_features.cuda()
        # print('\tnode_features', node_features)
        # print('\t node_features.shape', node_features.shape)

        ############################
        # 2. Build edge features
        ############################
        # edge_features = self.g.edata[GNN_EDGE_LABELS_KEY]
        # print('\t GNN_EDGE_TYPES_KEY.shape', self.g.edata[GNN_EDGE_TYPES_KEY].size())
        # print('\t GNN_EDGE_LABELS_KEY.shape', self.g.edata[GNN_EDGE_LABELS_KEY].size())
        edge_features = torch.cat((self.g.edata[GNN_EDGE_TYPES_KEY], self.g.edata[GNN_EDGE_LABELS_KEY]), dim=1)
               
        # edge_features = edge_features.view(edge_features.size()[0], -1)
        # self.edge_dim = edge_features.size()[1]
        if self.is_cuda:
            edge_features = edge_features.cuda()
        # print('\tedge_features', edge_features)
        # print('\t edge_features.shape', edge_features.shape)

        #################################
        # 4. Iterate over each layer
        #################################
        for layer_idx, layer in enumerate(self.edgnn_layers):
            if layer_idx == 0:  # these are gat layers
                h = node_features
            # else:
                # h = self.g.ndata['h_'+str(layer_idx-1)]
            # h = h.type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
            h = layer(h, edge_features, self.g)
            # save only last layer output
            if layer_idx == len(self.edgnn_layers)-1:
                key = 'h_' + str(layer_idx)
                self.g.ndata[key] = h
            
        #############################################################
        # 5. It's graph classification, construct readout function
        #############################################################
        # sum with weights so that only features of last nodes is used
        last_layer_key = 'h_' + str(len(self.edgnn_layers)-1)
        sum_node = dgl.sum_nodes(g, last_layer_key)
        sum_node = sum_node.type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
        # print('\t h_last_node', h_last_node)
        # print('\t h_last_node.shape', h_last_node.shape)

        final_output = self.fc(sum_node)
        # final_output = final_output.type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
        # print('\t final_output.shape', final_output.shape)
        # print('\n')
        
        return final_output


    def eval_node_classification(self, labels, mask):
        self.eval()
        loss_fcn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = self(None)
            logits = logits[mask]
            labels = labels[mask]
            loss = loss_fcn(logits, labels)
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels), loss

    def eval_graph_classification(self, labels, testing_graphs):
        self.eval()
        loss_fcn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = self(testing_graphs)
            loss = loss_fcn(logits, labels)
            _, indices = torch.max(logits, dim=1)
            corrects = torch.sum(indices == labels)
            
            return corrects.item() * 1.0 / len(labels), loss, logits
