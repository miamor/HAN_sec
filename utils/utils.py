from __future__ import print_function

import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp

import dill as pickle

import torch
from utils.constants import GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY, GNN_NODE_FEAT_OUT_KEY, GNN_EDGE_FEAT_KEY, GNN_AGG_MSG_KEY


interesting_args = ["heap_dep_bypass", "filepath_r", "filepath"]

# def getInterestingArg(args):
#     for key in args:
#         if key in interesting_args:

def care_APIs():
    return [
    "NtDuplicateObject",
    "DeviceIoControl",
    "MoveFileWithProgressTransactedW",
    "OpenServiceA",
    "NtQuerySystemInformation",
    "NtSetValueKey",
    "WNetGetProviderNameW",
    "NtSetInformationFile",
    "NtCreateProcessEx",
    "NtCreateKey",
    "RtlCreateUserProcess",
    "MoveFileWithProgressW",
    "CryptExportKey",
    "OpenServiceW",
    "NtOpenProcess",
    "ControlService",
    "CryptEncrypt",
    "NtTerminateProcess",
    "NtClose",
    "GetAdaptersAddresses",
    "CryptHashData",
    "RegQueryValueExW",
    "GetClipboardData",
    "Process32NextW",
    "RegSetValueExA",
    "CreateServiceA",
    "RegOpenKeyExW",
    "NtDelayExecution",
    "NtDeviceIoControlFile",
    "SetClipboardViewer",
    "NtAllocateVirtualMemory",
    "ReadProcessMemory",
    "RegOpenKeyExA",
    "ShellExecuteExW",
    "NtWriteFile",
    "LdrGetDllHandle",
    "CryptGenKey",
    "CreateServiceW",
    "GetComputerNameW",
    "RegQueryValueExA",
    "NtOpenFile",
    "InternetReadFile",
    "ObtainUserAgentString",
    "URLDownloadToCacheFileW",
    "GetUserNameA",
    "NtCreateFile",
    "AddClipboardFormatListener",
    "GetComputerNameA",
    "NtLoadDriver",
    "NtCreateProcess",
    "NtProtectVirtualMemory",
    "EnumServicesStatusA",
    "RegSetValueExW",
    "InternetSetOptionA",
    "SetWindowsHookExA",
    "LdrGetProcedureAddress",
    "SetWindowsHookExW",
    "EnumServicesStatusW",
    "Process32FirstW",
    "SetFileAttributesW",
    "InternetOpenA",
    "LdrLoadDll",
    "NtCreateUserProcess",
    "InternetOpenW",
    "CreateProcessInternalW",
    "URLDownloadToFileW"
]

def reset_graph_features(g):
    keys = [GNN_NODE_FEAT_IN_KEY, GNN_AGG_MSG_KEY, GNN_MSG_KEY, GNN_NODE_FEAT_OUT_KEY]
    for key in keys:
        if key in g.ndata:
            del g.ndata[key]
    if GNN_EDGE_FEAT_KEY in g.edata:
        del g.edata[GNN_EDGE_FEAT_KEY] 


def compute_node_degrees(g):
    """
    Given a graph, compute the degree of each node
    :param g: DGL graph
    :return: node_degrees: a tensor with the degree of each node
             node_degrees_ids: a labeled version of node_degrees (usable for 1-hot encoding)
    """
    fc = lambda i: g.in_degrees(i).item()
    node_degrees = list(map(fc, range(g.number_of_nodes())))
    unique_deg = list(set(node_degrees))
    mapping = dict(zip(unique_deg, list(range(len(unique_deg)))))
    node_degree_ids = [mapping[deg] for deg in node_degrees]
    return torch.LongTensor(node_degrees), torch.LongTensor(node_degree_ids)


def save_txt(obj, ofpath):
    """
    Save an object as text

    Args:
        obj (list): list to be converted to string to save to text
        ofpath (str): path where to store the file
    """
    with open(ofpath, 'w+') as ofh:
        ofh.write('\n'.join(obj))


def save_pickle(obj, ofpath):
    """
    Save an object as pickle

    Args:
        graph (DGLGraph): graph to be saved
        ofpath (str): path where to store the file
    """
    with open(ofpath, 'wb') as ofh:
        pickle.dump(obj, ofh)


def load_pickle(ifpath):
    """
    Load an object from pickle

    Args:
        ifpath (str): path from where a graph is loaded
    """
    with open(ifpath, 'rb') as ifh:
        return pickle.load(ifh)




def indices_to_one_hot(data, out_vec_size):
    """
    Convert an iterable of indices to one-hot encoded labels.
    """
    targets = np.array(data).reshape(-1)
    return np.eye(out_vec_size)[targets].reshape(-1)


def label_encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[
        i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj
