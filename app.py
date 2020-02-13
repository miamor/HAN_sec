import time
import numpy as np
import dgl
import torch
from torch.utils.data import DataLoader
import random
import json

from utils.early_stopping import EarlyStopping
from utils.io import load_checkpoint
from utils.utils import label_encode_onehot, indices_to_one_hot

from utils.constants import *
from models.model_edgnn import Model

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from utils.utils import load_pickle, save_pickle, save_txt

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

    def __init__(self, data, model_config, learning_config, pretrained_weight, early_stopping=True, patience=100, json_path=None, vocab_path=None, odir=None):
        self.data = data
        self.model_config = model_config
        # max length of a sequence (max nodes among graphs)
        self.seq_max_length = data[MAX_N_NODES]
        self.learning_config = learning_config
        self.pretrained_weight = pretrained_weight

        with open(vocab_path+'/../mapping.json', 'r') as f:
            self.mapping = json.load(f)

        self.labels = self.data[LABELS]
        self.graphs_names = self.data[GNAMES]

        data_graph = self.data[GRAPH]
        data_nclasses = self.data[N_CLASSES]
        if N_RELS in self.data:
            data_nrels = self.data[N_RELS]
        else:
            data_nrels = None
            
        if N_ENTITIES in self.data:
            data_nentities = self.data[N_ENTITIES]
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
        graphs = self.data[GRAPH]                 # load all the graphs

        # debug purposes: reshuffle all the data before the splitting
        random_indices = list(range(len(graphs)))
        random.shuffle(random_indices)
        graphs = [graphs[i] for i in random_indices]
        labels = self.labels[random_indices]
        graphs_names = [self.graphs_names[i] for i in random_indices]

        # Split train and test
        train_size = int(self.TRAIN_SIZE * len(graphs))
        g_train = graphs[:train_size]
        l_train = labels[:train_size]
        n_train = graphs_names[:train_size]

        g_test = graphs[train_size:]
        l_test = labels[train_size:]
        n_test = graphs_names[train_size:]

        if not os.path.isdir(self.odir):
            os.makedirs(self.odir)
        save_pickle(g_train, os.path.join(self.odir, 'train'))
        save_pickle(l_train, os.path.join(self.odir, 'train_labels'))
        save_pickle(g_test, os.path.join(self.odir, 'test'))
        save_pickle(l_test, os.path.join(self.odir, 'test_labels'))

        # save graph name list to txt file
        save_txt(n_train, os.path.join(self.odir, 'train_list.txt'))
        save_txt(n_test, os.path.join(self.odir, 'test_list.txt'))


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

    def test(self, model_path=''):
        print('Test model')
        
        try:
            print('*** Load pre-trained model '+model_path+' ***')
            self.model = load_checkpoint(self.model, model_path)
        except ValueError as e:
            print('Error while loading the model.', e)

        print('\nTest all')
        # acc = np.mean(self.accuracies)
        # acc = self.accuracies
        graphs = self.data[GRAPH]
        labels = self.labels
        self.run_test(graphs, labels)
                    
        print('\nTest on train graphs')
        graphs = load_pickle(os.path.join(self.odir, 'train'))
        labels = load_pickle(os.path.join(self.odir, 'train_labels'))
        self.run_test(graphs, labels)

        print('\nTest on test graphs')
        graphs = load_pickle(os.path.join(self.odir, 'test'))
        labels = load_pickle(os.path.join(self.odir, 'test_labels'))
        self.run_test(graphs, labels)


    def test_on_data(self, model_path=''):
        print('Test model')
        
        try:
            print('*** Load pre-trained model '+model_path+' ***')
            self.model = load_checkpoint(self.model, model_path)
        except ValueError as e:
            print('Error while loading the model.', e)

        print('\nTest on data')
        # acc = np.mean(self.accuracies)
        # acc = self.accuracies
        graphs = self.data[GRAPH]
        labels = self.labels
        self.run_test(graphs, labels)


    def run_test(self, graphs, labels):
        batches = dgl.batch(graphs)
        acc, _, logits = self.model.eval_graph_classification(labels, batches)
        _, indices = torch.max(logits, dim=1)
        labels = labels.cpu()
        indices = indices.cpu()
        # print('labels', labels)
        # print('indices', indices)
        # labels_txt = ['malware', 'benign']
            
        cm = confusion_matrix(y_true=labels, y_pred=indices)
        print(cm)
        print('Total samples', len(labels))
        
        if len(self.mapping) == 2:
            # lbl_mal = 1
            # lbl_bng = 0
            lbl_mal = self.mapping['malware']
            lbl_bng = self.mapping['benign']
            n_mal = (labels == lbl_mal).sum().item()
            n_bgn = (labels == lbl_bng).sum().item()
            tpr = cm[lbl_mal][lbl_mal]/n_mal * 100 # actual malware that is correctly detected as malware
            far = cm[lbl_bng][lbl_mal]/n_bgn * 100  # benign that is incorrectly labeled as malware
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
