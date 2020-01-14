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

    def __init__(self, data, model_config, learning_config, pretrained_weight, early_stopping=True, patience=100):
        self.data = data
        self.model_config = model_config
        # max length of a sequence (max nodes among graphs)
        self.seq_max_length = data[MAX_N_NODES]
        self.learning_config = learning_config
        self.pretrained_weight = pretrained_weight

        self.labels = self.data[LABELS]

        self.model = Model(g=data[GRAPH],
                           config_params=model_config,
                           n_classes=data[N_CLASSES],
                           n_rels=data[N_RELS] if N_RELS in data else None,
                           n_entities=data[N_ENTITIES] if N_ENTITIES in data else None,
                           is_cuda=learning_config['cuda'],
                           seq_dim=self.seq_max_length,
                           batch_size=1)

        if early_stopping:
            self.early_stopping = EarlyStopping(
                patience=patience, verbose=True)

    def train(self, save_path='', k_fold=10):
        if self.pretrained_weight is not None:
            self.model = load_checkpoint(self.model, self.pretrained_weight)

        loss_fcn = torch.nn.CrossEntropyLoss()

        # initialize graphs
        self.accuracies = np.zeros(10)
        graphs = self.data[GRAPH]                 # load all the graphs

        # debug purposes: reshuffle all the data before the splitting
        random_indices = list(range(len(graphs)))
        random.shuffle(random_indices)
        graphs = [graphs[i] for i in random_indices]
        labels = self.labels[random_indices]

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

            start = int(len(graphs)/K) * k
            end = int(len(graphs)/K) * (k+1)
            print('\n\n\nProcess new k, '+str(start)+'-'+str(end))

            # testing batch
            testing_graphs = graphs[start:end]
            self.testing_labels = labels[start:end]
            self.testing_batch = dgl.batch(testing_graphs)

            # training batch
            training_graphs = graphs[:start] + graphs[end:]
            training_labels = labels[list(
                range(0, start)) + list(range(end+1, len(graphs)))]
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
            print('self.testing_labels', self.testing_labels)

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

                val_acc, val_loss = self.model.eval_graph_classification(
                    self.testing_labels, self.testing_batch)
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

        # acc = np.mean(self.accuracies)
        # acc = self.accuracies
        graphs = self.data[GRAPH]

        labels = self.labels
        batches = dgl.batch(graphs)

        # print('\nbg', bg)
        acc, _ = self.model.eval_graph_classification(labels, batches)

        print("Test Accuracy {:.4f}".format(acc))
        
        # acc = np.mean(self.accuracies)

        return acc
