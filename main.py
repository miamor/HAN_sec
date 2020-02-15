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
import time
import shutil


def load_dataset(args, cuda):
    prep_data = PrepareData(reports_parent_dir_path=args.input_report_folder, data_json_path=args.input_data_file,
                            pickle_folder=args.input_data_folder, vocab_path=args.vocab_path, encode_edge_data=args.encode_edge_data, save_json=args.save_json, use_interesting_apis=args.use_interesting_apis,
                            prepend_vocab=args.prepend_vocab,
                            mapping_path=args.mapping_path)
    data = prep_data.load_data(from_folder=args.from_report_folder,
                               from_json=args.from_data_json,
                               from_pickle=args.from_pickle)
    data = to_cuda(data) if cuda else data
    return data


def run_app(args, data, cuda):
    # print_graph_stats(data[GRAPH])

    print('*** Load model from', args.config_fpath)

    ###########################
    # 1. Training
    ###########################
    if args.action == "train":
        now = time.strftime("%Y-%m-%d_%H-%M-%S")

        config_params = read_params(args.config_fpath, verbose=True)

        odir = 'output/'+now
        # default_path = create_default_path(odir+'/checkpoints')
        default_path = create_default_path(odir)
        print('\n*** Set default saving/loading path to:', default_path)

        learning_config = {'lr': args.lr, 'epochs': args.epochs,
                           'weight_decay': args.weight_decay, 'batch_size': args.batch_size, 'cuda': cuda}
        app = App(data, model_config=config_params[0], learning_config=learning_config,
                  pretrained_weight=args.checkpoint_file, early_stopping=True, patience=50, json_path=args.input_data_file, vocab_path=args.vocab_path, odir=odir)
        print('\n*** Start training ***\n')
        ''' save config to output '''
        shutil.copy(src=args.config_fpath, dst=odir+'/'+args.config_fpath.split('/')[-1])
        ''' train '''
        app.train(default_path, k_fold=args.k_fold, split_train_test=args.split_train_test)
        app.test(default_path)
        # remove_model(default_path)

    ###########################
    # 2. Testing
    ###########################
    # if args.action == "test" and args.checkpoint_file is not None:
    if args.action == "test":
        print('\n*** Start testing ***\n')
        learning_config = {'cuda': cuda}
        # odir = 'output/2020-01-14_15-04-01'
        odir = args.out_dir

        config_fpath = odir+'/config_edGNN_graph_class.json'
        config_params = read_params(config_fpath, verbose=True)

        if args.checkpoint_file is None:
            args.checkpoint_file = odir+'/checkpoint'

        app = App(data, model_config=config_params[0], learning_config=learning_config,
                  pretrained_weight=args.checkpoint_file, early_stopping=True, patience=80, json_path=args.input_data_file, vocab_path=args.vocab_path, odir=odir)
        app.test(args.checkpoint_file)


def run_app_2(args, data, cuda):
    # config_params = read_params(args.config_fpath, verbose=True)
    odir = args.out_dir
    config_fpath = odir+'/config_edGNN_graph_class.json'
    print('*** Load model from', config_fpath)
    config_params = read_params(config_fpath, verbose=True)

    if args.checkpoint_file is None:
        args.checkpoint_file = odir+'/checkpoint'

    print('\n*** Start testing ***\n')
    learning_config = {'cuda': cuda}

    app = App(data, model_config=config_params[0], learning_config=learning_config,
              pretrained_weight=args.checkpoint_file, early_stopping=True, patience=80, json_path=args.input_data_file, vocab_path=args.vocab_path)
    app.test_on_data(args.checkpoint_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--input_report_folder", default=None)
    parser.add_argument("-d", "--input_data_file", default=None)
    parser.add_argument("-p", "--input_data_folder", default=None)
    parser.add_argument("-f", "--folder", default=None)
    parser.add_argument("-c", "--config_fpath",
                        default='models/config/config_edGNN_graph_class.json')
    parser.add_argument("-v", "--vocab_path", default=None)
    parser.add_argument("-a", "--use_interesting_apis", type=bool, default=False)
    parser.add_argument("-pv", "--prepend_vocab", type=bool, default=False)
    parser.add_argument("-m", "--mapping_path", default=None)


    parser.add_argument("-fr", "--from_report_folder",
                        type=bool, default=False)
    parser.add_argument("-fd", "--from_data_json", type=bool, default=False)
    parser.add_argument("-fp", "--from_pickle", type=bool, default=False)

    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu")

    parser.add_argument("action", choices={'train', 'test', 'test_data', 'prep'})

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
    parser.add_argument("-o", "--out_dir", default=None)

    parser.add_argument("-sj", "--save_json", type=bool, default=True)
    parser.add_argument("-s", "--split_train_test", type=bool, default=False)
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
    data_ = load_dataset(args, cuda)

    ###########################
    # Run the app
    ###########################
    if args.action == "test_data":
        run_app_2(args, data_, cuda)
    else:
        run_app(args, data_,cuda)