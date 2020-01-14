import argparse
import numpy as np

import torch

# from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.regularizers import l2
# from keras.layers import Input, Dropout

# from keras.layers import Dense, Embedding, LSTM, Bidirectional, Activation, Flatten, Conv1D, TimeDistributed, Add, RepeatVector, Permute, Multiply, GRU

# from models import GraphAttention

from utils.prep_data import PrepareData
from utils.inits import to_cuda
from utils.io import print_graph_stats, read_params, create_default_path, remove_model

from app import App

from utils.constants import *


def load_dataset(args, cuda):
    prep_data = PrepareData(reports_parent_dir_path=args.input_report_folder, data_json_path=args.input_data_file,
                            pickle_folder=args.input_data_folder, vocab_path=args.vocab_path, encode_edge_data=args.encode_edge_data, save_json=args.save_json)
    data = prep_data.load_data(from_folder=args.from_report_folder,
                               from_json=args.from_data_json,
                               from_pickle=args.from_pickle)
    data = to_cuda(data) if cuda else data
    return data


def run_app(args, cuda):
    # print_graph_stats(data[GRAPH])

    default_path = create_default_path()
    print('\n*** Set default saving/loading path to:', default_path)

    config_params = read_params(args.config_fpath, verbose=True)

    ###########################
    # 1. Training
    ###########################
    if args.action == "train":
        learning_config = {'lr': args.lr, 'epochs': args.epochs,
                           'weight_decay': args.weight_decay, 'batch_size': args.batch_size, 'cuda': cuda}
        app = App(data, model_config=config_params[0], learning_config=learning_config,
                  pretrained_weight=args.checkpoint_file, early_stopping=True, patience=20)
        print('\n*** Start training ***\n')
        app.train(default_path, k_fold=args.k_fold)
        app.test(default_path)

    ###########################
    # 2. Testing
    ###########################
    if args.action == "test" and args.checkpoint_file is not None:
        print('\n*** Start testing ***\n')
        learning_config = {'cuda': cuda}
        app = App(data, model_config=config_params[0], learning_config=learning_config,
                  pretrained_weight=args.checkpoint_file, early_stopping=True, patience=20)
        app.test(args.checkpoint_file)

    ###########################
    # 3. Delete model
    ###########################
    # remove_model(default_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--input_report_folder",
                        default='/home/fitmta/Documents/MinhTu/Dataset/cuckoo_sm')
    parser.add_argument("-d", "--input_data_file", default='data/data.json')
    parser.add_argument("-p", "--input_data_folder", default='data/pickle')
    parser.add_argument("-f", "--folder", default=None)
    parser.add_argument("-o", "--out_folder", default=None)
    parser.add_argument("-c", "--config_fpath",
                        default='models/config/config_edGNN_graph_class.json')
    parser.add_argument("-v", "--vocab_path", default='data/vocab.txt')

    parser.add_argument("-fr", "--from_report_folder",
                        type=bool, default=False)
    parser.add_argument("-fd", "--from_data_json", type=bool, default=False)
    parser.add_argument("-fp", "--from_pickle", type=bool, default=False)

    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu")

    parser.add_argument("action", choices={'train', 'test', 'prep'})

    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size (only for graph classification)")
    parser.add_argument("--k_fold", type=int, default=10,
                        help="k_fold (only for graph classification)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="number of training epochs")
    parser.add_argument("--weight_decay", type=float,
                        default=5e-4, help="Weight for L2 loss")

    # parser.add_argument('test', action='store_true', default=False)
    parser.add_argument("-cp", "--checkpoint_file", default=None)

    parser.add_argument("-sj", "--save_json", type=bool, default=True)
    parser.add_argument("--encode_edge_data", type=bool, default=True)

    args = parser.parse_args()
    # print(args)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)

    ###########################
    # Load data
    ###########################
    data = load_dataset(args, cuda)

    ###########################
    # Run the app
    ###########################
    run_app(args, cuda)
