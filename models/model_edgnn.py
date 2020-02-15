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

import json
from utils.utils import preprocess_adj, care_APIs

from models.layers.edgnn import edGNNLayer
from models.layers.rgcn import RGCNLayer

ACTIVATIONS = {
    'relu': F.relu
}


node_type_code = {
        'proc': 0, # process_handle
        'file': 1, # file_handle
        'reg': 2, # registry key_handle

        # 'network': 2,
        'process_api': 3,
        'file_api': 4,
        'reg_api': 5,
}
interesting_apis = care_APIs() + list(node_type_code.keys()) + ['Other']


class Model(nn.Module):

    word_dict_node = []
    word_dict_edge = []

    word_to_ix = {}

    def __init__(self, g, config_params, n_classes=None, n_rels=None, n_entities=None, is_cuda=False, seq_dim=None, batch_size=1, json_path=None, vocab_path=None):
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
        
        self.embedding_dim = 5
        # self.json_data = {
        #     'nodes': {},
        #     'paths': {}
        # }
        # self.json_filenames = ['nodes', 'proc_process', 'proc_file', 'proc_reg', 'proc_network']
        # if json_path is not None: # read data
        #     for key in self.json_filenames:
        #             with open(json_path+'/'+key+'.json') as json_file:
        #                 if key == 'nodes':
        #                     self.json_data['nodes'] = json.load(json_file)
        #                 else:
        #                     self.json_data['paths'].update(json.load(json_file))


        self.vocab_path_node = vocab_path+'/node.txt'
        self.vocab_path_edge = vocab_path+'/edge.txt'
        self.get_word_dict()

        # layer_type = config_params["layer_type"]

        self.build_model()


    def get_word_dict(self):
        # read from dict node
        with open(self.vocab_path_node, 'r') as f:
            vocab = f.read().strip()
            self.word_dict_node = vocab.split(' ')
            # self.word_to_ix_node = {word: i for i,
            #                    word in enumerate(self.word_dict_node)}
            # self.num_token_node = len(self.word_dict_node)
        
        # read from dict edge
        with open(self.vocab_path_edge, 'r') as f:
            vocab = f.read().strip()
            self.word_dict_edge = vocab.split(' ')
            # self.word_to_ix_edge = {word: i for i,
            #                    word in enumerate(self.word_dict_edge)}
            # self.num_token_edge = len(self.word_dict_edge)

        ''' Combine '''
        self.word_dict = self.word_dict_node + self.word_dict_edge
        # self.word_to_ix = {word: i for i,
        #                        word in enumerate(self.word_dict)}
        self.num_token = len(self.word_dict)


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
        # print('self.g[0].edata', self.g[0].edata)
        # self.node_dim = self.g[0].ndata[GNN_NODE_TYPES_KEY].shape[1] + self.g[0].ndata[GNN_NODE_LABELS_KEY].shape[1]

        ''' because self.g[0].ndata[GNN_NODE_LABELS_KEY] will be passed through Embedding layer '''
        self.node_dim = self.g[0].ndata[GNN_NODE_TYPES_KEY].shape[1] + self.g[0].ndata[GNN_NODE_LABELS_KEY].shape[1] * self.embedding_dim
        self.edge_dim = self.g[0].edata[GNN_EDGE_TYPES_KEY].shape[1] + self.g[0].edata[GNN_EDGE_LABELS_KEY].shape[1] * self.embedding_dim
        # self.edge_dim = self.g[0].edata[GNN_EDGE_TYPES_KEY].shape[1] + self.g[0].edata[GNN_EDGE_LABELS_KEY].shape[1] * self.embedding_dim + self.g[0].edata[GNN_EDGE_BUFFER_SIZE_KEY].shape[1]

        print('self.node_dim, self.edge_dim', self.node_dim, self.edge_dim)

        """ Embedding layer """
        self.emb_layer = nn.Embedding(self.num_token, self.embedding_dim)
        print('* Embedding:', self.num_token, self.embedding_dim)

        """ edGNN layers """
        n_edGNN_layers = len(layer_params['n_units'])
        for i in range(n_edGNN_layers):
            if i == 0:  # take input from GAT layer
                print('* GNN:', self.node_dim, self.edge_dim,
                      layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]])

                edGNN = edGNNLayer(self.g, self.node_dim, self.edge_dim,
                                   layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]], is_cuda=self.is_cuda)
            else:
                print('* GNN:', layer_params['n_units'][i-1], self.edge_dim,
                      layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]])

                edGNN = edGNNLayer(self.g, layer_params['n_units'][i-1], self.edge_dim,
                                   layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]], is_cuda=self.is_cuda)

            # self.add_module('edGNN_{}'.format(i), edGNN)
            self.edgnn_layers.append(edGNN)
        
        """ Classification layer """
        print('* Building fc:', layer_params['n_units'][-1], self.n_classes)
        self.fc = nn.Linear(layer_params['n_units'][-1], self.n_classes)

        print('*** Model successfully built ***\n')


    def forward(self, g):
        # print(g)

        # if g is not None:
        #     g.set_n_initializer(dgl.init.zero_initializer)
        #     g.set_e_initializer(dgl.init.zero_initializer)
        #     self.g = g
        self.g = g

        ############################
        # 1. Build node features
        ############################
        # print('\t GNN_NODE_LABELS_KEY', self.g.ndata[GNN_NODE_LABELS_KEY])
        # print('\t GNN_NODE_TYPES_KEY.shape', self.g.ndata[GNN_NODE_TYPES_KEY].size())
        # print('\t GNN_NODE_LABELS_KEY.shape', self.g.ndata[GNN_NODE_LABELS_KEY].size())
        # node_features = self.g.ndata[GNN_NODE_LABELS_KEY]
        
        # ''' self.g.ndata[GNN_NODE_LABELS_KEY] is just the id of the node, not actually label. Now we need to read the label of the node from file! '''
        node_embed = self.emb_layer(self.g.ndata[GNN_NODE_LABELS_KEY]).view(self.g.ndata[GNN_NODE_TYPES_KEY].shape[0], -1)
        node_embed = node_embed.type(self.g.ndata[GNN_NODE_TYPES_KEY].type())

        # print('node_embed', node_embed)
        # print('node_embed.shape', node_embed.shape)
        # print('self.g.ndata[GNN_NODE_TYPES_KEY].shape', self.g.ndata[GNN_NODE_TYPES_KEY].shape)
        node_features = torch.cat((self.g.ndata[GNN_NODE_TYPES_KEY], node_embed), dim=1)

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
        # print('\t self.g.edata', self.g.edata)
        # print('\t self.g', self.g)
        # print('\t GNN_EDGE_TYPES_KEY', self.g.edata[GNN_EDGE_TYPES_KEY])
        # print('\t GNN_EDGE_LABELS_KEY', self.g.edata[GNN_EDGE_LABELS_KEY])
        # print('\t GNN_EDGE_BUFFER_SIZE_KEY', self.g.edata[GNN_EDGE_BUFFER_SIZE_KEY])

        # print('\t GNN_EDGE_TYPES_KEY.shape', self.g.edata[GNN_EDGE_TYPES_KEY].size())
        # print('\t GNN_EDGE_LABELS_KEY.shape', self.g.edata[GNN_EDGE_LABELS_KEY].size())
        # print('\t GNN_EDGE_BUFFER_SIZE_KEY.shape', self.g.edata[GNN_EDGE_BUFFER_SIZE_KEY].size())

        edge_embed = self.emb_layer(self.g.edata[GNN_EDGE_LABELS_KEY]).view(self.g.edata[GNN_EDGE_TYPES_KEY].shape[0], -1)
        # edge_embed = edge_embed.type(self.g.edata[GNN_EDGE_TYPES_KEY].type())
        edge_ft_lbl = edge_embed.type(torch.FloatTensor)

        edge_ft_type = self.g.edata[GNN_EDGE_TYPES_KEY].type(torch.FloatTensor)

        # print('edge_ft_type.shape', edge_ft_type.shape)

        # edge_features = torch.cat((self.g.edata[GNN_EDGE_TYPES_KEY], edge_ft_type), dim=1)
        edge_ft_bufsize = self.g.edata[GNN_EDGE_BUFFER_SIZE_KEY].type(torch.FloatTensor)
        edge_ft_bufsize = edge_ft_bufsize.div(torch.max(edge_ft_bufsize))
        # print('edge_ft_lbl', edge_ft_type)
        # print('edge_ft_type', edge_ft_type)
        # print('edge_ft_bufsize', edge_ft_bufsize)
        # edge_features = torch.cat((edge_ft_lbl, edge_ft_type, edge_ft_bufsize), dim=1)
        edge_features = torch.cat((edge_ft_lbl, edge_ft_type), dim=1)

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
