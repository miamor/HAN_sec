'''
chia data nhu a dung, classified va none. Chia 70/30 o moi thu muc roi combine lai
'''

import time
import numpy as np
import dgl
import torch
from torch.utils.data import DataLoader
import random

from utils.early_stopping import EarlyStopping
from utils.io import load_checkpoint
from utils.utils import label_encode_onehot, indices_to_one_hot

from utils.constants import *
from models.model_edgnn import Model

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from utils.utils import load_pickle, save_pickle

# def collate(samples):
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels).cuda() if labels[0].is_cuda else torch.tensor(labels)

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class App:
    """
    App inference
    """
    
    TRAIN_SIZE = 0.7

    def __init__(self, datas, model_config, learning_config, pretrained_weight, early_stopping=True, patience=100, json_path=None, vocab_path=None, odir=None):
        self.data1 = datas[0]
        self.data2 = datas[1] #none
        self.model_config = model_config
        # max length of a sequence (max nodes among graphs)
        self.seq_max_length = max(self.data1[MAX_N_NODES], self.data2[MAX_N_NODES])
        self.learning_config = learning_config
        self.pretrained_weight = pretrained_weight

        self.labels1 = self.data1[LABELS]
        self.labels2 = self.data2[LABELS]

        data_graph = self.data1[GRAPH] + self.data2[GRAPH]
        data_nclasses = self.data1[N_CLASSES] + self.data2[N_CLASSES]
        if N_RELS in self.data1 and N_RELS in self.data2:
            data_nrels = self.data1[N_RELS] + self.data2[N_RELS]
        else:
            data_nrels = None
            
        if N_ENTITIES in self.data1 and N_ENTITIES in self.data2:
            data_nentities = self.data1[N_ENTITIES] + self.data2[N_ENTITIES]
        else:
            data_nentities = None
            
        self.model = Model(g=data_graph,
                           config_params=model_config,
                           n_classes=data_nclasses,
                           n_rels=data_nrels,
                           n_entities=data_nentities,
                           is_cuda=learning_config['cuda'],
                           seq_dim=self.seq_max_length,
                           batch_size=1,
                           json_path=json_path,
                           vocab_path=vocab_path)

        if early_stopping:
            self.early_stopping = EarlyStopping(
                patience=patience, verbose=True)
            
        # Output folder to save train / test data
        if odir is None:
            odir = 'output/'+time.strftime("%Y-%m-%d_%H-%M-%S")
        self.odir = odir

    def train(self, save_path='', k_fold=10):
        if self.pretrained_weight is not None:
            self.model = load_checkpoint(self.model, self.pretrained_weight)

        loss_fcn = torch.nn.CrossEntropyLoss()

        # initialize graphs
        self.accuracies = np.zeros(k_fold)
        graphs1 = self.data1[GRAPH]                 # load all the graphs

        # debug purposes: reshuffle all the data before the splitting
        random_indices = list(range(len(graphs1)))
        random.shuffle(random_indices)
        graphs1 = [graphs1[i] for i in random_indices]
        labels1 = self.labels1[random_indices]

        # Split train and test
        train_size1 = int(self.TRAIN_SIZE * len(graphs1))
        g_train1 = graphs1[:train_size1]
        g_test1 = graphs1[train_size1:]
        l_train1 = labels1[:train_size1]
        l_test1 = labels1[train_size1:]
        
        
        graphs2 = self.data2[GRAPH]                 # load all the graphs

        # debug purposes: reshuffle all the data before the splitting
        random_indices = list(range(len(graphs2)))
        random.shuffle(random_indices)
        graphs2 = [graphs2[i] for i in random_indices]
        labels2 = self.labels2[random_indices]

        # Split train and test
        train_size2 = int(self.TRAIN_SIZE * len(graphs2))
        g_train2 = graphs2[:train_size2]
        g_test2 = graphs2[train_size2:]
        l_train2 = labels2[:train_size2]
        l_test2 = labels2[train_size2:]
        
        
        g_train = g_train1 + g_train2
        l_train = torch.cat((l_train1, l_train2))
        g_test = g_test1 + g_test2
        l_test = torch.cat((l_test1, l_test2))

        print('\n g_train1', len(g_train1))
        print('\n g_train2', len(g_train2))
        print('\n g_train', len(g_train))
        print('\n g_test1', len(g_test1))
        print('\n g_test2', len(g_test2))
        print('\n g_test', len(g_test))
        
        
        if not os.path.isdir(self.odir):
            os.makedirs(self.odir)
        save_pickle(g_train, os.path.join(self.odir, 'train'))
        save_pickle(l_train, os.path.join(self.odir, 'train_labels'))
        save_pickle(g_test, os.path.join(self.odir, 'test'))
        save_pickle(l_test, os.path.join(self.odir, 'test_labels'))


        K = k_fold
        for k in range(K):                  # K-fold cross validation

            # create GNN model
            # self.model = Model(g=self.data[GRAPH],
            #                    config_params=self.model_config,
            #                    n_classes=self.data[N_CLASSES],
            #                    n_rels=self.data[N_RELS] if N_RELS in self.data else None,
            #                    n_entities=self.data[N_ENTITIES] if N_ENTITIES in self.data else None,
            #                    is_cuda=self.learning_config['cuda'],
            #                    seq_dim=self.seq_max_length,
            #                    batch_size=1)

            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_config['lr'],
                                         weight_decay=self.learning_config['weight_decay'])

            if self.learning_config['cuda']:
                self.model.cuda()

            start = int(len(g_train)/K) * k
            end = int(len(g_train)/K) * (k+1)
            print('\n\n\nProcess new k='+str(k)+' | '+str(start)+'-'+str(end))

            # testing batch
            testing_graphs = g_train[start:end]
            testing_labels = l_train[start:end]
            testing_batch = dgl.batch(testing_graphs)

            # training batch
            training_graphs = g_train[:start] + g_train[end:]
            training_labels = l_train[list(
                range(0, start)) + list(range(end+1, len(g_train)))]
            training_samples = list(
                map(list, zip(training_graphs, training_labels)))
            training_batches = DataLoader(training_samples,
                                          batch_size=self.learning_config['batch_size'],
                                          shuffle=True,
                                          collate_fn=collate)

            print('training_graphs size: ', len(training_graphs))
            print('training_batches size: ', len(training_batches))
            print('testing_graphs size: ', len(testing_graphs))
            print('training_batches', training_batches)
            print('self.testing_labels', testing_labels)
            
            dur = []
            for epoch in range(self.learning_config['epochs']):
                self.model.train()
                if epoch >= 3:
                    t0 = time.time()
                losses = []
                training_accuracies = []
                for iter_idx, (bg, label) in enumerate(training_batches):
                    logits = self.model(bg)
                    if self.learning_config['cuda']:
                        label = label.cuda()
                    loss = loss_fcn(logits, label)
                    losses.append(loss.item())
                    _, indices = torch.max(logits, dim=1)
                    correct = torch.sum(indices == label)
                    training_accuracies.append(
                        correct.item() * 1.0 / len(label))

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                val_acc, val_loss, _ = self.model.eval_graph_classification(
                    testing_labels, testing_batch)
                print("Epoch {:05d} | Time(s) {:.4f} | train_acc {:.4f} | train_loss {:.4f} | val_acc {:.4f} | val_loss {:.4f}".format(
                    epoch, np.mean(dur) if dur else 0, np.mean(training_accuracies), np.mean(losses), val_acc, val_loss))

                is_better = self.early_stopping(
                    val_loss, self.model, save_path)
                if is_better:
                    self.accuracies[k] = val_acc

                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

            self.early_stopping.reset()

    def test(self, load_path=''):
        print('Test model')
        
        try:
            print('*** Load pre-trained model ***')
            self.model = load_checkpoint(self.model, load_path)
        except ValueError as e:
            print('Error while loading the model.', e)

        print('\nTest all')
        # acc = np.mean(self.accuracies)
        # acc = self.accuracies
        print('len graphs', len(self.data1[GRAPH]))
        print('len graphs', len(self.data2[GRAPH]))
        graphs = self.data1[GRAPH] + self.data2[GRAPH]
        print('len graphs', len(graphs))
        labels = torch.cat((self.labels1, self.labels2))
        self.run_test(graphs, labels)
        
        print('\nTest on train graphs')
        graphs = load_pickle(os.path.join(self.odir, 'train'))
        labels = load_pickle(os.path.join(self.odir, 'train_labels'))
        self.run_test(graphs, labels)

        print('\nTest on test graphs')
        graphs = load_pickle(os.path.join(self.odir, 'test'))
        labels = load_pickle(os.path.join(self.odir, 'test_labels'))
        self.run_test(graphs, labels)


    def run_test(self, graphs, labels):
        batches = dgl.batch(graphs)
        acc, _, logits = self.model.eval_graph_classification(labels, batches)
        _, indices = torch.max(logits, dim=1)
        # print('labels', labels)
        # print('indices', indices)
        # labels_txt = ['malware', 'benign']
        labels = labels.cpu()
        indices = indices.cpu()
        
        cm = confusion_matrix(y_true=labels, y_pred=indices)
        print(cm)
        print('Total samples', len(labels))
        
        n_mal = (labels == 0).sum().item()
        n_bgn = (labels == 1).sum().item()
        tpr = cm[0][0]/n_mal * 100 # actual malware that is correctly detected as malware
        far = cm[1][0]/n_bgn * 100  # benign that is incorrectly labeled as malware
        print('TPR', tpr)
        print('FAR', far)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(cm)
        # plt.title('Confusion matrix of the classifier')
        # fig.colorbar(cax)
        # # ax.set_xticklabels([''] + labels)
        # # ax.set_yticklabels([''] + labels)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.show()


        print("Accuracy {:.4f}".format(acc))
        
        # acc = np.mean(self.accuracies)

        return acc
