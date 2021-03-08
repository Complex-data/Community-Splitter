# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os.path
import random
import persona
from absl import app
from absl import flags
import networkx as nx
import numpy
from six.moves import xrange
import persona2vec
import networkx as nx
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_score
from sklearn.manifold import spectral_embedding
import node2vec
from gensim.models import Word2Vec
#from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
import time
import os
import tensorflow as tf
from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_edges_directed
import pickle
from copy import deepcopy
from sklearn.metrics import accuracy_score
#import keras
#from keras.layers import Embedding, Reshape, Activation, Input, merge
#from keras.models import Sequential, Model
#import keras.backend as K
import warnings
import graph_gan
import argparse, logging
import numpy as np
import struc2vec
from gensim.models.word2vec import LineSentence
import graph
import line
from classify import Classifier, read_node_label
from linegraph import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
from scipy import sparse
import topdegree, maxdegree
import splitter
import networkx.algorithms.community.label_propagation as label_prop
import networkx.algorithms.community.modularity_max as modularity
import networkx.algorithms.components.connected as components
import ilfrs
warnings.filterwarnings("ignore")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=True):

    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0) # actual value (0 for negative)
        
    # Calculate scores
	#hstack函数：按顺序堆叠数组
    preds_all = np.hstack([preds_pos, preds_neg])

    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    median = np.median(preds_all)
    index_pos = preds_all > median
    # A higher median probability is an edge that does not exist, so neg "<="
    index_neg = preds_all <= median
    preds_all[index_pos] = 1
    preds_all[index_neg] = 0
    # ap_score = precision_score(labels_all, preds_all)
    acc_score=accuracy_score(labels_all, preds_all)
    
    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score, acc_score

# Return a list of tuples (node1, node2) for networkx link prediction evaluation
def get_ebunch(train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
 
    test_edges_list = test_edges.tolist() # convert to nested list
    test_edges_list = [tuple(node_pair) for node_pair in test_edges_list] # convert node-pairs to tuples
    test_edges_false_list = test_edges_false.tolist()
    test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]
    return (test_edges_list + test_edges_false_list)
#---------------------------------------------------------------------------AA
# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def adamic_adar_scores(g_train, train_test_split):
    if g_train.is_directed(): # Only works for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    
    aa_scores = {}

    # Calculate scores
    aa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.adamic_adar_index(g_train, ebunch=get_ebunch(train_test_split)): # (u, v) = node indices, p = Adamic-Adar index
        aa_matrix[u][v] = p
        aa_matrix[v][u] = p # make sure it's symmetric
    aa_matrix = aa_matrix / aa_matrix.max() # Normalize matrix

    runtime = time.time() - start_time
    aa_roc, aa_ap, aa_acc = get_roc_score(test_edges, test_edges_false, aa_matrix)

    aa_scores['test_roc'] = aa_roc
    # aa_scores['test_roc_curve'] = aa_roc_curve
    aa_scores['test_ap'] = aa_ap
    aa_scores['test_acc'] = aa_acc
    aa_scores['runtime'] = runtime
    return aa_scores

#-----------------------------------------------------------------------------------------JC
# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def jaccard_coefficient_scores(g_train, train_test_split):
    if g_train.is_directed(): # Jaccard coef only works for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    jc_scores = {}

    # Calculate scores
    jc_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.jaccard_coefficient(g_train, ebunch=get_ebunch(train_test_split)): # (u, v) = node indices, p = Jaccard coefficient
        jc_matrix[u][v] = p
        jc_matrix[v][u] = p # make sure it's symmetric
    jc_matrix = jc_matrix / jc_matrix.max() # Normalize matrix

    runtime = time.time() - start_time
    jc_roc, jc_ap, jc_acc = get_roc_score(test_edges, test_edges_false, jc_matrix)

    jc_scores['test_roc'] = jc_roc
    # jc_scores['test_roc_curve'] = jc_roc_curve
    jc_scores['test_acc'] = jc_acc
    jc_scores['test_ap'] = jc_ap
    jc_scores['runtime'] = runtime
    return jc_scores

#-----------------------------------------------------------------------------------------------PA
# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def preferential_attachment_scores(g_train, train_test_split):
    if g_train.is_directed(): # Only defined for undirected graphs
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    pa_scores = {}

    # Calculate scores
    pa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.preferential_attachment(g_train, ebunch=get_ebunch(train_test_split)): # (u, v) = node indices, p = Jaccard coefficient
        pa_matrix[u][v] = p
        pa_matrix[v][u] = p # make sure it's symmetric
    pa_matrix = pa_matrix / pa_matrix.max() # Normalize matrix

    runtime = time.time() - start_time
    pa_roc, pa_ap, pa_acc = get_roc_score(test_edges, test_edges_false, pa_matrix)

    pa_scores['test_roc'] = pa_roc
    # pa_scores['test_roc_curve'] = pa_roc_curve
    pa_scores['test_acc'] = pa_acc
    pa_scores['test_ap'] = pa_ap
    pa_scores['runtime'] = runtime
    return pa_scores

#----------------------------------------------------------------------------------------SC
# Input: train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def spectral_clustering_scores(train_test_split, random_state=0):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    sc_scores = {}

    # Perform spectral clustering link prediction
    spectral_emb = spectral_embedding(adj_train, n_components=16, random_state=random_state)
    sc_score_matrix = np.dot(spectral_emb, spectral_emb.T)

    runtime = time.time() - start_time
    sc_test_roc, sc_test_ap, sc_test_acc = get_roc_score(test_edges, test_edges_false, sc_score_matrix, apply_sigmoid=True)
    #sc_val_roc, sc_val_ap = get_roc_score(val_edges, val_edges_false, sc_score_matrix, apply_sigmoid=True)

    # Record scores
    sc_scores['test_roc'] = sc_test_roc
    # sc_scores['test_roc_curve'] = sc_test_roc_curve
    sc_scores['test_acc'] = sc_test_acc
    sc_scores['test_ap'] = sc_test_ap

    #sc_scores['val_roc'] = sc_val_roc
    # sc_scores['val_roc_curve'] = sc_val_roc_curve
    #sc_scores['val_ap'] = sc_val_ap

    sc_scores['runtime'] = runtime
    return sc_scores
#----------------------------------------------------------------------node2vec-------------------------------------
# Input: NetworkX training graph, train_test_split (from mask_test_edges), n2v hyperparameters
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def node2vec_scores(
    g_train, train_test_split,
    P = 1, # Return hyperparameter
    Q = 1, # In-out hyperparameter
    WINDOW_SIZE = 10, # Context size for optimization
    NUM_WALKS = 10, # Number of walks per source
    WALK_LENGTH = 80, # Length of walk per source
    DIMENSIONS = 128, # Embedding dimension
    DIRECTED = False, # Graph directed/undirected
    WORKERS = 8, # Num. parallel workers
    ITER = 1, # SGD epochs
    edge_score_mode = "edge-emb", # Whether to use bootstrapped edge embeddings + LogReg (like in node2vec paper), 
        # or simple dot-product (like in GAE paper) for edge scoring
    verbose=1,
    ):
    if g_train.is_directed():
        DIRECTED = True

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    # Preprocessing, generate walks预处理,生成走
    if verbose >= 1:
        print ('Preprocessing grpah for node2vec...')
    g_n2v = node2vec.Graph(g_train, DIRECTED, P, Q) # create node2vec graph instance
    g_n2v.preprocess_transition_probs()
    if verbose == 2:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=True)
    else:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=False)
    walks = [map(str, walk) for walk in walks]

    # Train skip-gram model，训练skip-gram模型
    model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
    # Store embeddings mapping存储嵌入映射
    emb_mappings = model
    
    # Create node embeddings matrix (rows = nodes, columns = embedding features)
    #创建节点嵌入矩阵(行=节点，列=嵌入特性)
    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str] #the emb_mappings is a dic
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)


    # Generate bootstrapped edge embeddings (as is done in node2vec paper)生成引导的边缘嵌入(如node2vec文件中所做的那样)
        # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
        #生成引导的边缘嵌入(如node2vec文件中所做的那样)
    
    if edge_score_mode == "edge-emb":
        
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                #edge_emb = np.dot(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # Train-set edge embeddings，训练集边缘嵌入
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        #训练训练集边缘嵌入的逻辑回归分类器
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)
        
        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None
        
        n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        n2v_test_ap = average_precision_score(test_edge_labels, test_preds)
        median = np.median(test_preds)
        index_pos = test_preds > median
        index_neg = test_preds <= median
        test_preds[index_pos] = 1
        test_preds[index_neg] = 0
        n2v_test_acc = accuracy_score(test_edge_labels, test_preds)
        # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        

    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_acc']=n2v_test_acc
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['val_roc'] = n2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores

#-------------------------------------------------------Line---------------------------------------------
def linemethod(g_train,train_test_split):
    def parse_args():
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
        parser.add_argument('--clf_ratio', default=0.5, type=float,
                            help='The ratio of training data in the classification')
        parser.add_argument('--order', default=1, type=int,
                            help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
        parser.add_argument('--no-auto-save', action='store_true',
                            help='no save the best embeddings when training LINE')
        parser.add_argument('--epochs', default=10, type=int,
                            help='The training epochs of LINE')
        parser.add_argument('--representation_size', default=128, type=int,
                            help='dimenssion')
        parser.add_argument('--graph_format', default='edgelist',
                            help='Input graph format')
        parser.add_argument('--label_file',default=False)
        
        return parser.parse_args()

    start_time = time.time()
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split
    '''
    if args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)
    '''
    args = parse_args()
    g = LINEGraph()
    adj_train=adj_train.toarray()
    g.read_adj_to_graph(adj_train)

    
    model = line.LINE(g, epoch=args.epochs,
                              rep_size=args.representation_size, order=args.order)

    emb_matrix= model.save_embeddings_to_matrix()    

    def get_edge_embeddings(edge_list):
        embs = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = int(edge[1]-1)
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            #edge_emb = np.dot(emb1, emb2)
            embs.append(edge_emb)
        embs = np.array(embs)
        return embs

        # Train-set edge embeddings，训练集边缘嵌入
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        pos_val_edge_embs = get_edge_embeddings(val_edges)
        neg_val_edge_embs = get_edge_embeddings(val_edges_false)
        val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
        val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        #训练训练集边缘嵌入的逻辑回归分类器
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)
        
        # Predicted edge scores: probability of being of class "1" (real edge)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    runtime = time.time() - start_time

        # Calculate scores
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        line_val_roc = roc_auc_score(val_edge_labels, val_preds)
        # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        line_val_ap = average_precision_score(val_edge_labels, val_preds)
    else:
        line_val_roc = None
        line_val_roc_curve = None
        line_val_ap = None
        
    line_test_roc = roc_auc_score(test_edge_labels, test_preds)
    line_test_ap = average_precision_score(test_edge_labels, test_preds)
    median = np.median(test_preds)
    index_pos = test_preds >= median
    index_neg = test_preds < median
    test_preds[index_pos] = 1
    test_preds[index_neg] = 0
    line_test_acc = accuracy_score(test_edge_labels, test_preds)

    line_scores = {}
    line_scores['test_roc'] = line_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    line_scores['test_acc']=line_test_acc
    line_scores['test_ap'] = line_test_ap

    line_scores['runtime'] = runtime

    return line_scores
    
#----------------------------------------------------------Graph_gan---------------------------------------------
            
def Graph_gan_score(g_train,train_test_split):
    
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split
    n_node=adj_train.shape[0]

    start_time = time.time()
    
    
    graphgan = graph_gan.GraphGAN(g_train,n_node)
    emd=graphgan.train()
    
    runtime = time.time() - start_time
    
    test_edges=np.concatenate((test_edges,test_edges_false),axis=0)
    score_res=[]
    for i in range(len(test_edges)):
        score_res.append(np.dot(emd[test_edges[i][0]], emd[test_edges[i][1]]))
    test_label = np.array(score_res)

    true_label = np.zeros(test_label.shape)
    true_label[0: len(true_label) // 2] = 1

    graph_gan_test_roc = roc_auc_score(true_label, test_label)
    graph_gan_test_ap = average_precision_score(true_label, test_label)
    median = np.median(test_label)
    index_pos = test_label >= median
    index_neg = test_label < median
    test_label[index_pos] = 1
    test_label[index_neg] = 0
    graph_gan_test_acc = accuracy_score(true_label, test_label)
   
    graph_gan_scores = {}
    graph_gan_scores['test_roc'] = graph_gan_test_roc
    graph_gan_scores['test_acc'] = graph_gan_test_acc
    graph_gan_scores['test_ap'] = graph_gan_test_ap
    graph_gan_scores['runtime'] = runtime

    return graph_gan_scores

#----------------------------------------------------------------Struc2vec---------------------------------------------
def Structovec_score(train_test_split):
    logging.basicConfig(filename='struc2vec.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')

   
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    def parse_args():
        parser = argparse.ArgumentParser(description="Run struc2vec.")
        parser.add_argument('--input', nargs='?', default='/home/hxm/link_prediciton/struc2vec-master/graph/brazil-airports.edgelist',
                            help='Input graph path')
        parser.add_argument('--output', nargs='?', default='/home/hxm/link_prediciton/struc2vec-master/emb/brazil-airports.emb',
                            help='Embeddings path')
        parser.add_argument('--dimensions', type=int, default=128,
                            help='Number of dimensions. Default is 128.')
        parser.add_argument('--walk-length', type=int, default=80,
                            help='Length of walk per source. Default is 80.')
        parser.add_argument('--num-walks', type=int, default=10,
                            help='Number of walks per source. Default is 10.')
        parser.add_argument('--window-size', type=int, default=10,
                            help='Context size for optimization. Default is 10.')
        parser.add_argument('--until-layer', type=int, default=5,
                            help='Calculation until the layer.')
        parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD')
        parser.add_argument('--workers', type=int, default=8,
                            help='Number of parallel workers. Default is 8.')
        parser.add_argument('--weighted', dest='weighted', action='store_true',
                            help='Boolean specifying (un)weighted. Default is unweighted.')
        parser.add_argument('--unweighted', dest='unweighted', action='store_false')
        parser.set_defaults(weighted=False)
        parser.add_argument('--directed', dest='directed', action='store_true',
                            help='Graph is (un)directed. Default is undirected.')
        parser.add_argument('--undirected', dest='undirected', action='store_false')
        parser.set_defaults(directed=False)
        parser.add_argument('--OPT1', default=True, type=bool,
                        help='optimization 1')
        parser.add_argument('--OPT2', default=False, type=bool,
                        help='optimization 2')
        parser.add_argument('--OPT3', default=False, type=bool,
                        help='optimization 3')	
        return parser.parse_args()
    
    def read_graph():

        logging.info(" - Loading graph...")
        G = graph.load_edgelist(args.input,undirected=True)
        logging.info(" - Graph loaded.")
        return G
    
    def exec_struc2vec(args, graph_train):
        '''
        Pipeline for representational learning for all nodes in a graph.
        '''
        if(args.OPT3):
            until_layer = args.until_layer
        else:
            until_layer = None

        #G = read_graph()
        G = graph_train
        
        G = struc2vec.Graph(G, args.directed, args.workers, untilLayer = until_layer)

        if(args.OPT1):
            G.preprocess_neighbors_with_bfs_compact()
        else:
            G.preprocess_neighbors_with_bfs()

        if(args.OPT2):
            G.create_vectors()
            G.calc_distances(compactDegree = args.OPT1)
        else:
            G.calc_distances_all_vertices(compactDegree = args.OPT1)

        G.create_distances_network()
        G.preprocess_parameters_random_walk()
        G.simulate_walks(args.num_walks, args.walk_length)
        return G
    
    #graph_adj=adj_train.todense()

    
    graph_adj=adj_train.toarray()
 
    args = parse_args()
    graph_train= graph.Graph()
    for node_key in range( adj_train.shape[0]): 
        for node_value in range( adj_train.shape[0]):
            value_list= []
            if graph_adj[node_key][node_value]==1:
                value_list.append(node_value)
        graph_train[node_key]= value_list
            
    G = exec_struc2vec(args, graph_train)
    walks = LineSentence('/home/hxm/link_prediciton/random_walks.txt')
    model_struc = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1, workers=args.workers, iter=args.iter)
    emb_mappings = model_struc
    
    
    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)

    def get_edge_embeddings(edge_list):
        embs = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            #edge_emb = np.dot(emb1, emb2)
            embs.append(edge_emb)
        embs = np.array(embs)
        return embs

        # Train-set edge embeddings，训练集边缘嵌入
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        pos_val_edge_embs = get_edge_embeddings(val_edges)
        neg_val_edge_embs = get_edge_embeddings(val_edges_false)
        val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
        val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        #训练训练集边缘嵌入的逻辑回归分类器
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)
        
        # Predicted edge scores: probability of being of class "1" (real edge)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    runtime = time.time() - start_time

        # Calculate scores
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        s2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
        # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        s2v_val_ap = average_precision_score(val_edge_labels, val_preds)
    else:
        s2v_val_roc = None
        s2v_val_roc_curve = None
        s2v_val_ap = None
        
    s2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
    s2v_test_ap = average_precision_score(test_edge_labels, test_preds)
    median = np.median(test_preds)
    index_pos = test_preds >= median
    index_neg = test_preds < median
    test_preds[index_pos] = 1
    test_preds[index_neg] = 0
    s2v_test_acc = accuracy_score(test_edge_labels, test_preds)

    s2v_scores = {}
    s2v_scores['test_roc'] = s2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    s2v_scores['test_acc']=s2v_test_acc
    s2v_scores['test_ap'] = s2v_test_ap

    s2v_scores['val_roc'] = s2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    s2v_scores['val_ap'] = s2v_val_ap

    s2v_scores['runtime'] = runtime

    return s2v_scores

#---------------------------------------------------------------------------------------maxdegree
def maxdegree_scores(
    g_train, train_test_split,
    P = 1, # Return hyperparameter
    Q = 1, # In-out hyperparameter
    WINDOW_SIZE = 10, # Context size for optimization
    NUM_WALKS = 10, # Number of walks per source
    WALK_LENGTH = 80, # Length of walk per source
    DIMENSIONS = 128, # Embedding dimension
    DIRECTED = False, # Graph directed/undirected
    WORKERS = 8, # Num. parallel workers
    ITER = 1, # SGD epochs
    edge_score_mode = "edge-emb", # Whether to use bootstrapped edge embeddings + LogReg 
    verbose=1,
    ):
    if g_train.is_directed():
        DIRECTED = True

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    g_maxdeg = maxdegree.Graph(g_train, DIRECTED, P, Q) # create  graph instance
    #g_n2v.preprocess_transition_probs()
    if verbose == 2:
        walks = g_maxdeg.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=True)
    else:
        walks = g_maxdeg.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=False)
    walks = [map(str, walk) for walk in walks]

    # Train skip-gram model，训练skip-gram模型
    model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
    
    # Store embeddings mapping存储嵌入映射
    emb_mappings = model
    
    # Create node embeddings matrix (rows = nodes, columns = embedding features)
    #创建节点嵌入矩阵(行=节点，列=嵌入特性)
    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)

    # Generate bootstrapped edge embeddings (as is done in node2vec paper)生成引导的边缘嵌入(如node2vec文件中所做的那样)
        # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
        #生成引导的边缘嵌入(如node2vec文件中所做的那样)
    
    if edge_score_mode == "edge-emb":
        
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                #edge_emb = np.dot(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # Train-set edge embeddings，训练集边缘嵌入
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        #训练训练集边缘嵌入的逻辑回归分类器
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)
        
        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            maxdeg_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # maxdeg_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            maxdeg_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            maxdeg_val_roc = None
            maxdeg_val_roc_curve = None
            maxdeg_val_ap = None
        
        maxdeg_test_roc = roc_auc_score(test_edge_labels, test_preds)
        maxdeg_test_ap = average_precision_score(test_edge_labels, test_preds)
        median = np.median(test_preds)
        index_pos = test_preds >= median
        index_neg = test_preds < median
        test_preds[index_pos] = 1
        test_preds[index_neg] = 0
        maxdeg_test_acc = accuracy_score(test_edge_labels, test_preds)
        # maxdeg_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        

    # Record scores
    maxdeg_scores = {}

    maxdeg_scores['test_roc'] = maxdeg_test_roc
    # maxdeg_scores['test_roc_curve'] = maxdeg_test_roc_curve
    maxdeg_scores['test_acc']=maxdeg_test_acc
    maxdeg_scores['test_ap'] = maxdeg_test_ap

    maxdeg_scores['val_roc'] = maxdeg_val_roc
    # maxdeg_scores['val_roc_curve'] = maxdeg_val_roc_curve
    maxdeg_scores['val_ap'] = maxdeg_val_ap

    maxdeg_scores['runtime'] = runtime

    return maxdeg_scores
#--------------------------------------------------------------topdegree

def topdegree_scores(
    g_train, train_test_split,
    P = 1, # Return hyperparameter
    Q = 1, # In-out hyperparameter
    WINDOW_SIZE = 10, # Context size for optimization
    NUM_WALKS = 10, # Number of walks per source
    WALK_LENGTH = 80, # Length of walk per source
    DIMENSIONS = 128, # Embedding dimension
    DIRECTED = False, # Graph directed/undirected
    WORKERS = 8, # Num. parallel workers
    ITER = 1, # SGD epochs
    edge_score_mode = "edge-emb", # Whether to use bootstrapped edge embeddings + LogReg 
    verbose=1,
    ):
    if g_train.is_directed():
        DIRECTED = True

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    g_topdeg = topdegree.Graph(g_train, DIRECTED, P, Q) # create  graph instance
    #g_n2v.preprocess_transition_probs()
    if verbose == 2:
        walks = g_topdeg.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=True)
    else:
        walks = g_topdeg.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=False)
    walks = [map(str, walk) for walk in walks]

    # Train skip-gram model，训练skip-gram模型
    model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
    
    # Store embeddings mapping存储嵌入映射
    emb_mappings = model
    
    # Create node embeddings matrix (rows = nodes, columns = embedding features)
    #创建节点嵌入矩阵(行=节点，列=嵌入特性)
    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]

        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)


    # Generate bootstrapped edge embeddings (as is done in node2vec paper)生成引导的边缘嵌入(如node2vec文件中所做的那样)
        # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
        #生成引导的边缘嵌入(如node2vec文件中所做的那样)
    
    if edge_score_mode == "edge-emb":
        
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                #edge_emb = np.dot(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # Train-set edge embeddings，训练集边缘嵌入
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        #训练训练集边缘嵌入的逻辑回归分类器
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)
        
        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            topdeg_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # topdeg_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            topdeg_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            topdeg_val_roc = None
            topdeg_val_roc_curve = None
            topdeg_val_ap = None
        
        topdeg_test_roc = roc_auc_score(test_edge_labels, test_preds)
        topdeg_test_ap = average_precision_score(test_edge_labels, test_preds)
        median = np.median(test_preds)
        index_pos = test_preds >= median
        index_neg = test_preds < median
        test_preds[index_pos] = 1
        test_preds[index_neg] = 0
        topdeg_test_acc = accuracy_score(test_edge_labels, test_preds)
        # topdeg_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        
    # Record scores
    topdeg_scores = {}
    topdeg_scores['test_roc'] = topdeg_test_roc
    # topdeg_scores['test_roc_curve'] = topdeg_test_roc_curve
    topdeg_scores['test_acc']=topdeg_test_acc
    topdeg_scores['test_ap'] = topdeg_test_ap
    topdeg_scores['val_roc'] = topdeg_val_roc
    # topdeg_scores['val_roc_curve'] = topdeg_val_roc_curve
    topdeg_scores['val_ap'] = topdeg_val_ap
    topdeg_scores['runtime'] = runtime
    return topdeg_scores

#--------------------------------------------------------------splitter


def splitter_scores(g_name, edgelist_path, train_test_split,embedding_dim=64,
             walk_length=40,
             num_walks_node=10,
             constraint_learning_rate_scaling_factor=0.1,
             iterations=1,
             seed=1,
             window_size=5,
             local_clustering_fn=label_prop.label_propagation_communities):

    output_persona_embedding='/home/hxm/persona/64_splitter/'+g_name+'_persona_emb.embedding'
    output_embedding_prior='/home/hxm/persona/64_splitter/'+g_name+'_deepwalk.embedding'
    output_persona_graph_mapping='/home/hxm/persona/64_splitter/'+g_name+'_persona_map.txt'
    output_persona_graph='/home/hxm/persona/64_splitter/'+g_name+'_persona_graph.txt'

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
    start_time = time.time()
    graph = nx.read_edgelist(edgelist_path, create_using=nx.Graph)


    #local_clustering_fn = label_prop.label_propagation_communities
    #local_clustering_fn = persona._CLUSTERING_FN['modularity']
    # read persona args
    #local_clustering_fn = persona._CLUSTERING_FN[FLAGS.local_clustering_method]
    print('Running splitter...')
    res_splitter = splitter.Splitter(
        graph,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        num_walks_node=num_walks_node,
        constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
        iterations=iterations,
        seed=seed,
        window_size=window_size,
        local_clustering_fn=local_clustering_fn)

    # output embeddings
    res_splitter['persona_model'].save_word2vec_format(
        open(output_persona_embedding, 'wb'))

    # optional outputF
    if output_embedding_prior is not None:
        res_splitter['regular_model'].save_word2vec_format(
            open(output_embedding_prior, 'wb'))

    if output_persona_graph is not None:
        nx.write_edgelist(res_splitter['persona_graph'], output_persona_graph)

    if output_persona_graph_mapping is not None:
        with open(output_persona_graph_mapping, 'w') as outfile:
            for persona_node, original_node in res_splitter['persona_id_mapping'].items():
                outfile.write('{} {}\n'.format(persona_node, original_node))

    modell=Word2Vec()
    model=modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum=open(output_persona_embedding,'r')
    firstline=fnum.readline()
    line=firstline.split()
    num=int(line[0])+1
    col=int(line[1])
    fnum.close()

    emb_list = []
    nullemb=[0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb=nullemb
        emb_list.append(node_emb)
    p_emb_matrix = np.vstack(emb_list)


    # optional output
    persona_mapping={}

    for persona_node, original_node in res_splitter['persona_id_mapping'].items():
        if original_node not in persona_mapping:
            persona_mapping[original_node]=[int(persona_node)]
        else:
            persona_mapping[original_node]+=[int(persona_node)]

    for i in persona_mapping.keys():
        persona_mapping[i]+=[int(i)]


    def get_edge_embeddings(edge_list):
        resembs=[]

        for edge in edge_list:

            node1 = edge[0]
            node2 = edge[1]
            emb1=[]
            emb2=[]
            emb1.append(list(p_emb_matrix[node1]))
            emb2.append(list(p_emb_matrix[node2]))
            max_emb=np.dot(p_emb_matrix[node1],p_emb_matrix[node2])
#            cemb1=list(p_emb_matrix[node1])
#            cemb2=list(p_emb_matrix[node2])
            for i in persona_mapping[str(node1)]:
                emb1.append(list(p_emb_matrix[i]))
            emb1= np.vstack(emb1)


            for j in persona_mapping[str(node2)]:
                emb2.append(list(p_emb_matrix[j]))
            emb2= np.vstack(emb2)


            for node1emb in range(len(emb1)):
                for node2emb in range(len(emb2)):
                    edge_emb = np.dot(emb1[node1emb], emb2[node2emb])
                    if edge_emb>max_emb:
                        max_emb=edge_emb
#                        cemb1=emb1[node1emb]
#                        cemb2=emb2[node2emb]

#            emb=np.multiply(cemb1,cemb2)
#            resembs.append(emb)
            resembs.append(max_emb)
        resembs = np.array(resembs)

        return resembs

 # Train-set edge embeddings，训练集边缘嵌入
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
    train_edge_embs=train_edge_embs.reshape(-1, 1)

    # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    train_edge_labels=train_edge_labels.reshape(-1, 1)

    # Val-set edge embeddings, labels
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        pos_val_edge_embs = get_edge_embeddings(val_edges)
        neg_val_edge_embs = get_edge_embeddings(val_edges_false)
        val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
        val_edge_embs=val_edge_embs.reshape(-1, 1)
        val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
    test_edge_embs=test_edge_embs.reshape(-1, 1)

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    #训练训练集边缘嵌入的逻辑回归分类器
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    runtime = time.time() - start_time

    # Calculate scores
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        splitter_val_roc = roc_auc_score(val_edge_labels, val_preds)
        # splitter_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        splitter_val_ap = average_precision_score(val_edge_labels, val_preds)
    else:
        splitter_val_roc = None
        splitter_val_roc_curve = None
        splitter_val_ap = None
        
    splitter_test_roc = roc_auc_score(test_edge_labels, test_preds)
    splitter_test_ap = average_precision_score(test_edge_labels, test_preds)
    median = np.median(test_preds)
    index_pos = test_preds >= median
    index_neg = test_preds < median
    test_preds[index_pos] = 1
    test_preds[index_neg] = 0
    splitter_test_acc = accuracy_score(test_edge_labels, test_preds)

    splitter_scores = {}
    splitter_scores['test_roc'] = splitter_test_roc
    # topdeg_scores['test_roc_curve'] = topdeg_test_roc_curve
    splitter_scores['test_acc']=splitter_test_acc
    splitter_scores['test_ap'] = splitter_test_ap
    splitter_scores['val_roc'] = splitter_val_roc
    # topdeg_scores['val_roc_curve'] = topdeg_val_roc_curve
    splitter_scores['val_ap'] = splitter_val_ap
    splitter_scores['runtime'] = runtime
    return splitter_scores



def splitlstm_scores(g_name, edgelist_path, train_test_split,embedding_dim=64,
             walk_length=40,
             num_walks_node=10,
             constraint_learning_rate_scaling_factor=0.1,
             iterations=1,
             seed=1,
             window_size=5,
             local_clustering_fn=label_prop.label_propagation_communities):

    output_persona_embedding='/home/hxm/persona/64_splitter/'+g_name+'_persona_emb.embedding'
    output_embedding_prior='/home/hxm/persona/64_splitter/'+g_name+'_deepwalk.embedding'
    output_persona_graph_mapping='/home/hxm/persona/64_splitter/'+g_name+'_persona_map.txt'
    output_persona_graph='/home/hxm/persona/64_splitter/'+g_name+'_persona_graph.txt'

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
    start_time = time.time()


    modell=Word2Vec()
    model=modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum=open(output_persona_embedding,'r')
    firstline=fnum.readline()
    line=firstline.split()
    num=int(line[0])+1
    col=int(line[1])
    fnum.close()

    emb_list = []
    nullemb=[0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb=nullemb
        emb_list.append(node_emb)
    p_emb_matrix = np.vstack(emb_list)
    

    # optional output
    persona_mapping={}
    fmap=open(output_persona_graph_mapping,'r')
    for line in fmap.readlines():
        lines=line.split()
        if lines[1] not in persona_mapping:
            persona_mapping[lines[1]]=[int(lines[0])]
        else:
            persona_mapping[lines[1]]+=[int(lines[0])]
    fmap.close()

    for i in persona_mapping.keys():
        persona_mapping[i]+=[int(i)]

    
    embres=[]
    lstm=nn.LSTM(64,64)
    hidden=(torch.randn(1,1,64),torch.randn(1,1,64))

    for i in range(0,adj_train.shape[0]):

        perlstm=[]
        for perid in persona_mapping[str(i)]:
            perlstm.append([p_emb_matrix[perid]])
        inputs=torch.Tensor(perlstm)
        out,hidden=lstm(inputs,hidden)
        #res=(torch.max(out,0)[0]).detach().numpy()
        res=(torch.mean(out,0)).detach().numpy()
        ans=res[0]
        embres.append(ans)
    emb_mat = np.vstack(embres)
    emb_matrix=emb_mat*100

    np.save('/home/hxm/persona/emb/karate-splitlstm',emb_matrix)
    def get_edge_embeddings(edge_list):
        embs = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            #edge_emb = np.dot(emb1, emb2)
            edge_emb = np.multiply(emb1, emb2)
            embs.append(edge_emb)
        embs = np.array(embs)

        return embs

    
 # Train-set edge embeddings，训练集边缘嵌入
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])


    # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])
    

    # Val-set edge embeddings, labels
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        pos_val_edge_embs = get_edge_embeddings(val_edges)
        neg_val_edge_embs = get_edge_embeddings(val_edges_false)
        val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])

        val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])


    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    #训练训练集边缘嵌入的逻辑回归分类器
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    runtime = time.time() - start_time

    # Calculate scores
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        splitlstm_val_roc = roc_auc_score(val_edge_labels, val_preds)
        # splitlstm_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        splitlstm_val_ap = average_precision_score(val_edge_labels, val_preds)
    else:
        splitlstm_val_roc = None
        splitlstm_val_roc_curve = None
        splitlstm_val_ap = None
        
    splitlstm_test_roc = roc_auc_score(test_edge_labels, test_preds)
    splitlstm_test_ap = average_precision_score(test_edge_labels, test_preds)
    median = np.median(test_preds)
    index_pos = test_preds >= median
    index_neg = test_preds < median
    test_preds[index_pos] = 1
    test_preds[index_neg] = 0
    splitlstm_test_acc = accuracy_score(test_edge_labels, test_preds)

    splitlstm_scores = {}
    splitlstm_scores['test_roc'] = splitlstm_test_roc
    # splitlstm_scores['test_roc_curve'] = splitlstm_test_roc_curve
    splitlstm_scores['test_acc']=splitlstm_test_acc
    splitlstm_scores['test_ap'] = splitlstm_test_ap
    splitlstm_scores['val_roc'] = splitlstm_val_roc
    # splitlstm_scores['val_roc_curve'] = splitlstm_val_roc_curve
    splitlstm_scores['val_ap'] = splitlstm_val_ap
    splitlstm_scores['runtime'] = runtime
    return splitlstm_scores




def splitBIlstm_scores(g_name, edgelist_path, train_test_split,embedding_dim=64,
             walk_length=40,
             num_walks_node=10,
             constraint_learning_rate_scaling_factor=0.1,
             iterations=1,
             seed=1,
             window_size=5,
             num_layers=1,
             hidden_size=64,
             local_clustering_fn=label_prop.label_propagation_communities):

    output_persona_embedding='/home/hxm/persona/64_splitter/'+g_name+'_persona_emb.embedding'
    output_embedding_prior='/home/hxm/persona/64_splitter/'+g_name+'_deepwalk.embedding'
    output_persona_graph_mapping='/home/hxm/persona/64_splitter/'+g_name+'_persona_map.txt'
    output_persona_graph='/home/hxm/persona/64_splitter/'+g_name+'_persona_graph.txt'

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
    start_time = time.time()


    modell=Word2Vec()
    model=modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum=open(output_persona_embedding,'r')
    firstline=fnum.readline()
    line=firstline.split()
    num=int(line[0])+1
    col=int(line[1])
    fnum.close()

    emb_list = []
    nullemb=[0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb=nullemb
        emb_list.append(node_emb)
    p_emb_matrix = np.vstack(emb_list)
    

    # optional output
    persona_mapping={}
    fmap=open(output_persona_graph_mapping,'r')
    for line in fmap.readlines():
        lines=line.split()
        if lines[1] not in persona_mapping:
            persona_mapping[lines[1]]=[int(lines[0])]
        else:
            persona_mapping[lines[1]]+=[int(lines[0])]
    fmap.close()

    for i in persona_mapping.keys():
        persona_mapping[i]+=[int(i)]

    
    embres=[]
    lstm=nn.LSTM(embedding_dim,hidden_size,num_layers, bidirectional = True)
    hidden = (torch.randn(2*num_layers, 1, hidden_size), torch.randn(2*num_layers, 1, hidden_size))

    for i in range(0,adj_train.shape[0]):

        perlstm=[]
        for perid in persona_mapping[str(i)]:
            perlstm.append([p_emb_matrix[perid]])
        inputs=torch.Tensor(perlstm)
        out,hidden=lstm(inputs,hidden)
        #res=(torch.max(out,0)[0]).detach().numpy()
        res=(torch.mean(out,0)).detach().numpy()
        ans=res[0]
        embres.append(ans)
    emb_mat = np.vstack(embres)
    emb_matrix=emb_mat*100
    np.save('/home/hxm/persona/emb/karate-splitBIlstm',emb_matrix)
 
    def get_edge_embeddings(edge_list):
        embs = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            #edge_emb = np.dot(emb1, emb2)
            edge_emb = np.multiply(emb1, emb2)
            embs.append(edge_emb)
        embs = np.array(embs)

        return embs

    
 # Train-set edge embeddings，训练集边缘嵌入
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])


    # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])
    

    # Val-set edge embeddings, labels
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        pos_val_edge_embs = get_edge_embeddings(val_edges)
        neg_val_edge_embs = get_edge_embeddings(val_edges_false)
        val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])

        val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])


    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    #训练训练集边缘嵌入的逻辑回归分类器
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    runtime = time.time() - start_time

    # Calculate scores
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        splitBIlstm_val_roc = roc_auc_score(val_edge_labels, val_preds)
        # splitBIlstm_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        splitBIlstm_val_ap = average_precision_score(val_edge_labels, val_preds)
    else:
        splitBIlstm_val_roc = None
        splitBIlstm_val_roc_curve = None
        splitBIlstm_val_ap = None
        
    splitBIlstm_test_roc = roc_auc_score(test_edge_labels, test_preds)
    splitBIlstm_test_ap = average_precision_score(test_edge_labels, test_preds)
    median = np.median(test_preds)
    index_pos = test_preds >= median
    index_neg = test_preds < median
    test_preds[index_pos] = 1
    test_preds[index_neg] = 0
    splitBIlstm_test_acc = accuracy_score(test_edge_labels, test_preds)

    splitBIlstm_scores = {}
    splitBIlstm_scores['test_roc'] = splitBIlstm_test_roc
    # splitBIlstm_scores['test_roc_curve'] = splitBIlstm_test_roc_curve
    splitBIlstm_scores['test_acc']=splitBIlstm_test_acc
    splitBIlstm_scores['test_ap'] = splitBIlstm_test_ap
    splitBIlstm_scores['val_roc'] = splitBIlstm_val_roc
    # splitBIlstm_scores['val_roc_curve'] = splitBIlstm_val_roc_curve
    splitBIlstm_scores['val_ap'] = splitBIlstm_val_ap
    splitBIlstm_scores['runtime'] = runtime
    return splitBIlstm_scores





def splitgru_scores(g_name, edgelist_path, train_test_split,embedding_dim=64,
             walk_length=40,
             num_walks_node=10,
             constraint_learning_rate_scaling_factor=0.1,
             iterations=1,
             seed=1,
             window_size=5,
             local_clustering_fn=label_prop.label_propagation_communities):

    output_persona_embedding='/home/hxm/persona/64_splitter/'+g_name+'_persona_emb.embedding'
    output_embedding_prior='/home/hxm/persona/64_splitter/'+g_name+'_deepwalk.embedding'
    output_persona_graph_mapping='/home/hxm/persona/64_splitter/'+g_name+'_persona_map.txt'
    output_persona_graph='/home/hxm/persona/64_splitter/'+g_name+'_persona_graph.txt'

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
    start_time = time.time()


    modell=Word2Vec()
    model=modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum=open(output_persona_embedding,'r')
    firstline=fnum.readline()
    line=firstline.split()
    num=int(line[0])+1
    col=int(line[1])
    fnum.close()

    emb_list = []
    nullemb=[0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb=nullemb
        emb_list.append(node_emb)
    p_emb_matrix = np.vstack(emb_list)

    

    # optional output
    persona_mapping={}
    fmap=open(output_persona_graph_mapping,'r')
    for line in fmap.readlines():
        lines=line.split()
        if lines[1] not in persona_mapping:
            persona_mapping[lines[1]]=[int(lines[0])]
        else:
            persona_mapping[lines[1]]+=[int(lines[0])]
    fmap.close()

    for i in persona_mapping.keys():
        persona_mapping[i]+=[int(i)]

    
    embres=[]
    gru=nn.GRU(64,64,1, bidirectional = False)
    
    for i in range(0,adj_train.shape[0]):

        perlstm=[]
        for perid in persona_mapping[str(i)]:
            perlstm.append([p_emb_matrix[perid]])
        inputs=torch.Tensor(perlstm)
        out,hidden=gru(inputs,None)
        res=(torch.mean(out,0)).detach().numpy()
        ans=res[0]
        embres.append(ans)
    emb_mat = np.vstack(embres)
    emb_matrix=emb_mat*100
    np.save('/home/hxm/persona/emb/karate-splitgru',emb_matrix)
 
    def get_edge_embeddings(edge_list):
        embs = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            #edge_emb = np.dot(emb1, emb2)
            edge_emb = np.multiply(emb1, emb2)
            embs.append(edge_emb)
        embs = np.array(embs)

        return embs

    
 # Train-set edge embeddings，训练集边缘嵌入
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])


    # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])
    

    # Val-set edge embeddings, labels
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        pos_val_edge_embs = get_edge_embeddings(val_edges)
        neg_val_edge_embs = get_edge_embeddings(val_edges_false)
        val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])

        val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])


    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    #训练训练集边缘嵌入的逻辑回归分类器
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    runtime = time.time() - start_time

    # Calculate scores
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        splitgru_val_roc = roc_auc_score(val_edge_labels, val_preds)
        # splitgru_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        splitgru_val_ap = average_precision_score(val_edge_labels, val_preds)
    else:
        splitgru_val_roc = None
        splitgru_val_roc_curve = None
        splitgru_val_ap = None
        
    splitgru_test_roc = roc_auc_score(test_edge_labels, test_preds)
    splitgru_test_ap = average_precision_score(test_edge_labels, test_preds)
    median = np.median(test_preds)
    index_pos = test_preds >= median
    index_neg = test_preds < median
    test_preds[index_pos] = 1
    test_preds[index_neg] = 0
    splitgru_test_acc = accuracy_score(test_edge_labels, test_preds)

    splitgru_scores = {}
    splitgru_scores['test_roc'] = splitgru_test_roc
    # splitgru_scores['test_roc_curve'] = splitgru_test_roc_curve
    splitgru_scores['test_acc']=splitgru_test_acc
    splitgru_scores['test_ap'] = splitgru_test_ap
    splitgru_scores['val_roc'] = splitgru_val_roc
    # splitgru_scores['val_roc_curve'] = splitgru_val_roc_curve
    splitgru_scores['val_ap'] = splitgru_val_ap
    splitgru_scores['runtime'] = runtime
    return splitgru_scores



def splitBIgru_scores(g_name, edgelist_path, train_test_split,embedding_dim=64,
             walk_length=40,
             num_walks_node=10,
             constraint_learning_rate_scaling_factor=0.1,
             iterations=1,
             seed=1,
             window_size=5,
             num_layers=1,
             hidden_size=64,
             local_clustering_fn=label_prop.label_propagation_communities):

    output_persona_embedding='/home/hxm/persona/64_splitter/'+g_name+'_persona_emb.embedding'
    output_embedding_prior='/home/hxm/persona/64_splitter/'+g_name+'_deepwalk.embedding'
    output_persona_graph_mapping='/home/hxm/persona/64_splitter/'+g_name+'_persona_map.txt'
    output_persona_graph='/home/hxm/persona/64_splitter/'+g_name+'_persona_graph.txt'

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
    start_time = time.time()


    modell=Word2Vec()
    model=modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum=open(output_persona_embedding,'r')
    firstline=fnum.readline()
    line=firstline.split()
    num=int(line[0])+1
    col=int(line[1])
    fnum.close()

    emb_list = []
    nullemb=[0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb=nullemb
        emb_list.append(node_emb)
    p_emb_matrix = np.vstack(emb_list)

    # optional output
    persona_mapping={}
    fmap=open(output_persona_graph_mapping,'r')
    for line in fmap.readlines():
        lines=line.split()
        if lines[1] not in persona_mapping:
            persona_mapping[lines[1]]=[int(lines[0])]
        else:
            persona_mapping[lines[1]]+=[int(lines[0])]
    fmap.close()

    for i in persona_mapping.keys():
        persona_mapping[i]+=[int(i)]

    
    embres=[]
    gru=nn.GRU(embedding_dim,hidden_size,num_layers, bidirectional = True) #64 64 1
    hidden = torch.randn(2*num_layers, 1, hidden_size)
    
    for i in range(0,adj_train.shape[0]):

        perlstm=[]
        for perid in persona_mapping[str(i)]:
            perlstm.append([p_emb_matrix[perid]])
        inputs=torch.Tensor(perlstm)
        out,hidden=gru(inputs,hidden)
        res=(torch.mean(out,0)).detach().numpy()
        ans=res[0]
        embres.append(ans)
    emb_mat = np.vstack(embres)
    emb_matrix=emb_mat*100
    np.save('/home/hxm/persona/emb/karate-splitBIgru',emb_matrix)
 
    def get_edge_embeddings(edge_list):
        embs = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            #edge_emb = np.dot(emb1, emb2)
            edge_emb = np.multiply(emb1, emb2)
            embs.append(edge_emb)
        embs = np.array(embs)

        return embs

    
 # Train-set edge embeddings，训练集边缘嵌入
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])


    # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])
    

    # Val-set edge embeddings, labels
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        pos_val_edge_embs = get_edge_embeddings(val_edges)
        neg_val_edge_embs = get_edge_embeddings(val_edges_false)
        val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])

        val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])


    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    #训练训练集边缘嵌入的逻辑回归分类器
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    runtime = time.time() - start_time

    # Calculate scores
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        splitBIgru_val_roc = roc_auc_score(val_edge_labels, val_preds)
        # splitBIgru_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        splitBIgru_val_ap = average_precision_score(val_edge_labels, val_preds)
    else:
        splitBIgru_val_roc = None
        splitBIgru_val_roc_curve = None
        splitBIgru_val_ap = None
        
    splitBIgru_test_roc = roc_auc_score(test_edge_labels, test_preds)
    splitBIgru_test_ap = average_precision_score(test_edge_labels, test_preds)
    median = np.median(test_preds)
    index_pos = test_preds >= median
    index_neg = test_preds < median
    test_preds[index_pos] = 1
    test_preds[index_neg] = 0
    splitBIgru_test_acc = accuracy_score(test_edge_labels, test_preds)

    splitBIgru_scores = {}
    splitBIgru_scores['test_roc'] = splitBIgru_test_roc
    # splitBIgru_scores['test_roc_curve'] = splitBIgru_test_roc_curve
    splitBIgru_scores['test_acc']=splitBIgru_test_acc
    splitBIgru_scores['test_ap'] = splitBIgru_test_ap
    splitBIgru_scores['val_roc'] = splitBIgru_val_roc
    # splitBIgru_scores['val_roc_curve'] = splitBIgru_val_roc_curve
    splitBIgru_scores['val_ap'] = splitBIgru_val_ap
    splitBIgru_scores['runtime'] = runtime
    return splitBIgru_scores


def splitCommunity_scores(g_name, edgelist_path, train_test_split, embedding_dim=64,
                    walk_length=40,
                    num_walks_node=10,
                    constraint_learning_rate_scaling_factor=0.1,
                    iterations=1,
                    seed=1,
                    window_size=5,
                    num_layers=1,
                    hidden_size=64,
                    local_clustering_fn=label_prop.label_propagation_communities,
                    method='BIgru'):

    output_persona_embedding = '/home/hxm/persona/64_splitter/' + g_name + '_persona_emb.embedding'
    output_embedding_prior = '/home/hxm/persona/64_splitter/' + g_name + '_deepwalk.embedding'
    output_persona_graph_mapping = '/home/hxm/persona/64_splitter/' + g_name + '_persona_map.txt'
    output_persona_graph = '/home/hxm/persona/64_splitter/' + g_name + '_persona_graph.txt'

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
    start_time = time.time()
    graph = nx.read_edgelist(edgelist_path, create_using=nx.Graph)

    print('Running splitter...')
    res_splitter = splitter.Splitter(
        graph,
        g_name,     # 用来存每种社区划分的结果
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        num_walks_node=num_walks_node,
        constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
        iterations=iterations,
        seed=seed,
        window_size=window_size,
        local_clustering_fn=local_clustering_fn)

    # output embeddings
    res_splitter['persona_model'].save_word2vec_format(
        open(output_persona_embedding, 'wb'))

    # optional outputF
    if output_embedding_prior is not None:
        res_splitter['regular_model'].save_word2vec_format(
            open(output_embedding_prior, 'wb'))

    if output_persona_graph is not None:
        nx.write_edgelist(res_splitter['persona_graph'], output_persona_graph)

    if output_persona_graph_mapping is not None:
        with open(output_persona_graph_mapping, 'w') as outfile:
            for persona_node, original_node in res_splitter['persona_id_mapping'].items():
                outfile.write('{} {}\n'.format(persona_node, original_node))

    modell=Word2Vec()
    model=modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum=open(output_persona_embedding,'r')
    firstline=fnum.readline()
    line=firstline.split()
    num=int(line[0])+1
    col=int(line[1])
    fnum.close()

    emb_list = []
    nullemb=[0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb=nullemb
        emb_list.append(node_emb)
    p_emb_matrix = np.vstack(emb_list)

    # optional output
    persona_mapping={}


    for persona_node, original_node in res_splitter['persona_id_mapping'].items():
        if original_node not in persona_mapping:
            persona_mapping[original_node]=[int(persona_node)]
        else:
            persona_mapping[original_node]+=[int(persona_node)]

    for i in persona_mapping.keys():
        persona_mapping[i]+=[int(i)]
    '''
    if method == 'BIgru':
        embres=[]
        gru=nn.GRU(embedding_dim,hidden_size,num_layers, bidirectional = True)
        hidden = torch.randn(2 * num_layers, 1, hidden_size)

        for i in range(0, adj_train.shape[0]):

            perlstm = []
            for perid in persona_mapping[str(i)]:
                perlstm.append([p_emb_matrix[perid]])
            inputs = torch.Tensor(perlstm)
            out, hidden = gru(inputs, hidden)
            res = (torch.mean(out, 0)).detach().numpy()
            ans = res[0]
            embres.append(ans)
        emb_mat = np.vstack(embres)
        emb_matrix = emb_mat * 100
        np.save('/home/hxm/persona/emb/karate-splitCommunity', emb_matrix)
    else:
        embres=[]
        lstm=nn.LSTM(embedding_dim,hidden_size,num_layers, bidirectional = True)
        hidden = (torch.randn(2 * num_layers, 1, hidden_size), torch.randn(2 * num_layers, 1, hidden_size))

        for i in range(0, adj_train.shape[0]):

            perlstm = []
            for perid in persona_mapping[str(i)]:
                perlstm.append([p_emb_matrix[perid]])
            inputs = torch.Tensor(perlstm)
            out, hidden = lstm(inputs, hidden)
            res = (torch.mean(out, 0)).detach().numpy()
            ans = res[0]
            embres.append(ans)
        emb_mat = np.vstack(embres)
        emb_matrix = emb_mat * 100
        np.save('/home/hxm/persona/emb/karate-splitCommunity', emb_matrix)
    '''
    def get_edge_embeddings(edge_list):
        resembs=[]

        for edge in edge_list:

            node1 = edge[0]
            node2 = edge[1]
            emb1 = []
            emb2 = []
            emb1.append(list(p_emb_matrix[node1]))
            emb2.append(list(p_emb_matrix[node2]))
            max_emb = np.dot(p_emb_matrix[node1], p_emb_matrix[node2])
            #            cemb1=list(p_emb_matrix[node1])
            #            cemb2=list(p_emb_matrix[node2])
            for i in persona_mapping[str(node1)]:
                emb1.append(list(p_emb_matrix[i]))
            emb1 = np.vstack(emb1)

            for j in persona_mapping[str(node2)]:
                emb2.append(list(p_emb_matrix[j]))
            emb2 = np.vstack(emb2)

            for node1emb in range(len(emb1)):
                for node2emb in range(len(emb2)):
                    edge_emb = np.dot(emb1[node1emb], emb2[node2emb])
                    if edge_emb > max_emb:
                        max_emb = edge_emb
            #                        cemb1=emb1[node1emb]
            #                        cemb2=emb2[node2emb]

            #            emb=np.multiply(cemb1,cemb2)
            #            resembs.append(emb)
            resembs.append(max_emb)
        resembs = np.array(resembs)

        return resembs

    # Train-set edge embeddings，训练集边缘嵌入
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
    train_edge_embs = train_edge_embs.reshape(-1, 1)

    # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    # Val-set edge embeddings, labels
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        pos_val_edge_embs = get_edge_embeddings(val_edges)
        neg_val_edge_embs = get_edge_embeddings(val_edges_false)
        val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
        val_edge_embs = val_edge_embs.reshape(-1, 1)
        val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
    test_edge_embs = test_edge_embs.reshape(-1, 1)
    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    # 训练训练集边缘嵌入的逻辑回归分类器
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    runtime = time.time() - start_time

    # Calculate scores
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        splitCommunity_val_roc = roc_auc_score(val_edge_labels, val_preds)
        # splitgru_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        splitCommunity_val_ap = average_precision_score(val_edge_labels, val_preds)
    else:
        splitCommunity_val_roc = None
        splitgru_val_roc_curve = None
        splitCommunity_val_ap = None


    splitCommunity_test_roc = roc_auc_score(test_edge_labels, test_preds)
    splitCommunity_test_ap = average_precision_score(test_edge_labels, test_preds)
    median = np.median(test_preds)
    index_pos = test_preds >= median
    index_neg = test_preds < median
    test_preds[index_pos] = 1
    test_preds[index_neg] = 0
    splitCommunity_test_acc = accuracy_score(test_edge_labels, test_preds)

    splitCommunity_scores = {}
    splitCommunity_scores['test_roc'] = splitCommunity_test_roc
    # splitgru_scores['test_roc_curve'] = splitgru_test_roc_curve
    splitCommunity_scores['test_acc'] = splitCommunity_test_acc
    splitCommunity_scores['test_ap'] = splitCommunity_test_ap
    splitCommunity_scores['val_roc'] = splitCommunity_val_roc
    # splitgru_scores['val_roc_curve'] = splitgru_val_roc_curve
    splitCommunity_scores['val_ap'] = splitCommunity_val_ap
    splitCommunity_scores['runtime'] = runtime
    return splitCommunity_scores

def splitCommunity2_scores(g_name, edgelist_path, train_test_split, embedding_dim=64,
                    walk_length=40,
                    num_walks_node=10,
                    constraint_learning_rate_scaling_factor=0.1,
                    iterations=1,
                    seed=1,
                    window_size=5,
                    num_layers=1,
                    hidden_size=64,
                    local_clustering_fn=label_prop.label_propagation_communities,
                    method='BIgru'):
    output_persona_embedding = '/home/hxm/persona/64_splitter/' + g_name + '_persona_emb.embedding'
    output_embedding_prior = '/home/hxm/persona/64_splitter/' + g_name + '_deepwalk.embedding'
    output_persona_graph_mapping = '/home/hxm/persona/64_splitter/' + g_name + '_persona_map.txt'
    output_persona_graph = '/home/hxm/persona/64_splitter/' + g_name + '_persona_graph.txt'

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
    start_time = time.time()
    graph = nx.read_edgelist(edgelist_path, create_using=nx.Graph)


    # 社区划分1
    print('Running splitter...')
    res_splitter = splitter.Splitter(
        graph,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        num_walks_node=num_walks_node,
        constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
        iterations=iterations,
        seed=seed,
        window_size=window_size,
        local_clustering_fn=label_prop.label_propagation_communities)

    # output embeddings
    res_splitter['persona_model'].save_word2vec_format(
        open(output_persona_embedding, 'wb'))

    # optional outputF
    if output_embedding_prior is not None:
        res_splitter['regular_model'].save_word2vec_format(
            open(output_embedding_prior, 'wb'))

    if output_persona_graph is not None:
        nx.write_edgelist(res_splitter['persona_graph'], output_persona_graph)

    if output_persona_graph_mapping is not None:
        with open(output_persona_graph_mapping, 'w') as outfile:
            for persona_node, original_node in res_splitter['persona_id_mapping'].items():
                outfile.write('{} {}\n'.format(persona_node, original_node))

    modell=Word2Vec()
    model=modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum=open(output_persona_embedding,'r')
    firstline=fnum.readline()
    line=firstline.split()
    num=int(line[0])+1
    col=int(line[1])
    fnum.close()

    emb_list = []
    nullemb=[0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb=nullemb
        emb_list.append(node_emb)
    p_emb_matrix1 = np.vstack(emb_list)

    # optional output
    persona_mapping1={}


    for persona_node, original_node in res_splitter['persona_id_mapping'].items():
        if original_node not in persona_mapping1:
            persona_mapping1[original_node]=[int(persona_node)]
        else:
            persona_mapping1[original_node]+=[int(persona_node)]

    for i in persona_mapping1.keys():
        persona_mapping1[i]+=[int(i)]


    # 社区划分2
    print('Running splitter...')
    res_splitter = splitter.Splitter(
        graph,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        num_walks_node=num_walks_node,
        constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
        iterations=iterations,
        seed=seed,
        window_size=window_size,
        local_clustering_fn=components.connected_components)

    # output embeddings
    res_splitter['persona_model'].save_word2vec_format(
        open(output_persona_embedding, 'wb'))

    # optional outputF
    if output_embedding_prior is not None:
        res_splitter['regular_model'].save_word2vec_format(
            open(output_embedding_prior, 'wb'))

    if output_persona_graph is not None:
        nx.write_edgelist(res_splitter['persona_graph'], output_persona_graph)

    if output_persona_graph_mapping is not None:
        with open(output_persona_graph_mapping, 'w') as outfile:
            for persona_node, original_node in res_splitter['persona_id_mapping'].items():
                outfile.write('{} {}\n'.format(persona_node, original_node))

    modell=Word2Vec()
    model=modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum=open(output_persona_embedding,'r')
    firstline=fnum.readline()
    line=firstline.split()
    num=int(line[0])+1
    col=int(line[1])
    fnum.close()

    emb_list = []
    nullemb=[0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb=nullemb
        emb_list.append(node_emb)
    p_emb_matrix2 = np.vstack(emb_list)

    # optional output
    persona_mapping2={}


    for persona_node, original_node in res_splitter['persona_id_mapping'].items():
        if original_node not in persona_mapping2:
            persona_mapping2[original_node]=[int(persona_node)]
        else:
            persona_mapping2[original_node]+=[int(persona_node)]

    for i in persona_mapping2.keys():
        persona_mapping2[i]+=[int(i)]


    # 社区划分3
    print('Running splitter...')
    res_splitter = splitter.Splitter(
        graph,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        num_walks_node=num_walks_node,
        constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
        iterations=iterations,
        seed=seed,
        window_size=window_size,
        local_clustering_fn=modularity.greedy_modularity_communities)

    # output embeddings
    res_splitter['persona_model'].save_word2vec_format(
        open(output_persona_embedding, 'wb'))

    # optional outputF
    if output_embedding_prior is not None:
        res_splitter['regular_model'].save_word2vec_format(
            open(output_embedding_prior, 'wb'))

    if output_persona_graph is not None:
        nx.write_edgelist(res_splitter['persona_graph'], output_persona_graph)

    if output_persona_graph_mapping is not None:
        with open(output_persona_graph_mapping, 'w') as outfile:
            for persona_node, original_node in res_splitter['persona_id_mapping'].items():
                outfile.write('{} {}\n'.format(persona_node, original_node))

    modell=Word2Vec()
    model=modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum=open(output_persona_embedding,'r')
    firstline=fnum.readline()
    line=firstline.split()
    num=int(line[0])+1
    col=int(line[1])
    fnum.close()

    emb_list = []
    nullemb=[0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb=nullemb
        emb_list.append(node_emb)
    p_emb_matrix3 = np.vstack(emb_list)

    # optional output
    persona_mapping3={}


    for persona_node, original_node in res_splitter['persona_id_mapping'].items():
        if original_node not in persona_mapping3:
            persona_mapping3[original_node]=[int(persona_node)]
        else:
            persona_mapping3[original_node]+=[int(persona_node)]

    for i in persona_mapping3.keys():
        persona_mapping3[i]+=[int(i)]

    # 社区划分4
    print('Running splitter...')
    res_splitter = splitter.Splitter(
        graph,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        num_walks_node=num_walks_node,
        constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
        iterations=iterations,
        seed=seed,
        window_size=window_size,
        local_clustering_fn=ilfrs.ilfrs)

    # output embeddings
    res_splitter['persona_model'].save_word2vec_format(
        open(output_persona_embedding, 'wb'))

    # optional outputF
    if output_embedding_prior is not None:
        res_splitter['regular_model'].save_word2vec_format(
            open(output_embedding_prior, 'wb'))

    if output_persona_graph is not None:
        nx.write_edgelist(res_splitter['persona_graph'], output_persona_graph)

    if output_persona_graph_mapping is not None:
        with open(output_persona_graph_mapping, 'w') as outfile:
            for persona_node, original_node in res_splitter['persona_id_mapping'].items():
                outfile.write('{} {}\n'.format(persona_node, original_node))

    modell = Word2Vec()
    model = modell.load_word2vec_format(open(output_persona_embedding, 'rb'))
    fnum = open(output_persona_embedding, 'r')
    firstline = fnum.readline()
    line = firstline.split()
    num = int(line[0]) + 1
    col = int(line[1])
    fnum.close()

    emb_list = []
    nullemb = [0 for i in range(col)]
    for node_index in range(0, num):
        node_str = str(node_index)
        try:
            node_emb = model[node_str]
        except KeyError:
            node_emb = nullemb
        emb_list.append(node_emb)
    p_emb_matrix4 = np.vstack(emb_list)

    # optional output
    persona_mapping4 = {}

    for persona_node, original_node in res_splitter['persona_id_mapping'].items():
        if original_node not in persona_mapping4:
            persona_mapping4[original_node] = [int(persona_node)]
        else:
            persona_mapping4[original_node] += [int(persona_node)]

    for i in persona_mapping4.keys():
        persona_mapping4[i] += [int(i)]

    if method == 'BIgru':
        embres=[]
        gru=nn.GRU(embedding_dim,embedding_dim,num_layers, bidirectional = True)
        # hidden = torch.randn(2*num_layers, 1, hidden_size)
        for i in range(0, adj_train.shape[0]):

            perlstm = []

            for perid in persona_mapping1[str(i)]:
                perlstm.append([p_emb_matrix1[perid]])
            for perid in persona_mapping2[str(i)]:
                perlstm.append([p_emb_matrix2[perid]])
            for perid in persona_mapping3[str(i)]:
                perlstm.append([p_emb_matrix3[perid]])
            for perid in persona_mapping4[str(i)]:
                perlstm.append([p_emb_matrix4[perid]])
            inputs = torch.Tensor(perlstm)
            out, hidden = gru(inputs, None)
            res = (torch.mean(out, 0)).detach().numpy()
            ans = res[0]
            embres.append(ans)
        emb_mat = np.vstack(embres)
        emb_matrix = emb_mat * 100
        np.save('/home/hxm/persona/emb/karate-splitCommunity', emb_matrix)
    else:
        embres=[]
        lstm=nn.LSTM(embedding_dim,embedding_dim,num_layers, bidirectional = True)
        # hidden = (torch.randn(2 * num_layers, 1, hidden_size), torch.randn(2 * num_layers, 1, hidden_size))
        for i in range(0, adj_train.shape[0]):

            perlstm = []

            for perid in persona_mapping1[str(i)]:
                perlstm.append([p_emb_matrix1[perid]])
            for perid in persona_mapping2[str(i)]:
                perlstm.append([p_emb_matrix2[perid]])
            for perid in persona_mapping3[str(i)]:
                perlstm.append([p_emb_matrix3[perid]])
            for perid in persona_mapping4[str(i)]:
                perlstm.append([p_emb_matrix4[perid]])
            inputs = torch.Tensor(perlstm)
            out, hidden = lstm(inputs, None)
            res = (torch.mean(out, 0)).detach().numpy()
            ans = res[0]
            embres.append(ans)
        emb_mat = np.vstack(embres)
        emb_matrix = emb_mat * 100
        np.save('/home/hxm/persona/emb/karate-splitCommunity', emb_matrix)

    # 这里开始与persona无关
    def get_edge_embeddings(edge_list):
        embs = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            #edge_emb = np.dot(emb1, emb2)
            edge_emb = np.multiply(emb1, emb2)
            embs.append(edge_emb)
        embs = np.array(embs)

        return embs

    # Train-set edge embeddings，训练集边缘嵌入
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

    # Create train-set edge labels: 1 = real edge, 0 = false edge，创建训练集边缘标签:1 =实边缘，0 =假边缘
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    # Val-set edge embeddings, labels
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        pos_val_edge_embs = get_edge_embeddings(val_edges)
        neg_val_edge_embs = get_edge_embeddings(val_edges_false)
        val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])

        val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    # 训练训练集边缘嵌入的逻辑回归分类器
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    runtime = time.time() - start_time

    # Calculate scores
    if len(val_edges) > 0 and len(val_edges_false) > 0:
        splitCommunity_val_roc = roc_auc_score(val_edge_labels, val_preds)
        # splitgru_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        splitCommunity_val_ap = average_precision_score(val_edge_labels, val_preds)
    else:
        splitCommunity_val_roc = None
        splitgru_val_roc_curve = None
        splitCommunity_val_ap = None


    splitCommunity_test_roc = roc_auc_score(test_edge_labels, test_preds)
    splitCommunity_test_ap = average_precision_score(test_edge_labels, test_preds)
    median = np.median(test_preds)
    index_pos = test_preds >= median
    index_neg = test_preds < median
    test_preds[index_pos] = 1
    test_preds[index_neg] = 0
    splitCommunity_test_acc = accuracy_score(test_edge_labels, test_preds)

    splitCommunity_scores = {}
    splitCommunity_scores['test_roc'] = splitCommunity_test_roc
    # splitgru_scores['test_roc_curve'] = splitgru_test_roc_curve
    splitCommunity_scores['test_acc'] = splitCommunity_test_acc
    splitCommunity_scores['test_ap'] = splitCommunity_test_ap
    splitCommunity_scores['val_roc'] = splitCommunity_val_roc
    # splitgru_scores['val_roc_curve'] = splitgru_val_roc_curve
    splitCommunity_scores['val_ap'] = splitCommunity_val_ap
    splitCommunity_scores['runtime'] = runtime
    return splitCommunity_scores



# Input: adjacency matrix (in sparse format), features_matrix (normal format), test_frac, val_frac, verbose
    # Verbose: 0 - print nothing, 1 - print scores, 2 - print scores + GAE training progress
# Returns: Dictionary of results (ROC AUC, ROC Curve, AP, Runtime) for each link prediction method
def calculate_all_scores(g_name, adj_sparse, features_matrix=None, directed=False, \
        test_frac=.3, val_frac=.1, random_state=0, verbose=1, \
        train_test_split_file=None,
        tf_dtype=tf.float32):
    np.random.seed(random_state) # Guarantee consistent train/test split
    tf.set_random_seed(random_state)  # Consistent GAE training
    #tf.compat.v1.set_random_seed(np.random.randint(0, 10))

    # Prepare LP scores dictionary
    lp_scores = {}

    ### ---------- PREPROCESSING ---------- ###
    train_test_split = None
    try: # If found existing train-test split, use that file
        with open(train_test_split_file, 'rb') as f:
            train_test_split = pickle.load(f)
            print ('Found existing train-test split!')
    except: # Else, generate train-test split on the fly
        print ('Generating train-test split...')
        if directed == False:
            train_test_split = mask_test_edges(adj_sparse, test_frac=test_frac, val_frac=val_frac)
        else:
            train_test_split = mask_test_edges_directed(adj_sparse, test_frac=test_frac, val_frac=val_frac)
    
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack tuple
    '''
    sparse.save_npz('/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/adj_train', adj_train) 
    numpy_list_1 = np.asarray(train_edges)
    np.savetxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/train_edges", numpy_list_1,fmt='%d',delimiter=' ',newline='\r\n')
    numpy_list_2 = np.asarray(train_edges_false)
    np.savetxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/train_edges_false", numpy_list_2,fmt='%d',delimiter=' ',newline='\r\n')
    numpy_list_3 = np.asarray(val_edges)
    np.savetxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/val_edges", numpy_list_3,fmt='%d',delimiter=' ',newline='\r\n')
    numpy_list_4 = np.asarray(val_edges_false)
    np.savetxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/val_edges_false", numpy_list_4,fmt='%d',delimiter=' ',newline='\r\n')
    numpy_list_5 = np.asarray(test_edges)
    np.savetxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/test_edges", numpy_list_5,fmt='%d',delimiter=' ',newline='\r\n')
    numpy_list_6 = np.asarray(test_edges_false)
    np.savetxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/test_edges_false", numpy_list_6,fmt='%d',delimiter=' ',newline='\r\n')
    '''
    '''
    adj_train= sparse.load_npz('/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/adj_train.npz') 
    #numpy_list = np.asarray(train_edges)
    #np.savetxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/train_edges", numpy_list,fmt='%d',delimiter=' ',newline='\r\n')
    train_edges=np.loadtxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/train_edges")
    train_edges_false=np.loadtxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/train_edges_false")
    test_edges_false=np.loadtxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/test_edges_false")
    test_edges=np.loadtxt("/home/hxm/link_prediciton/train-test-splits/Wiki-Vote/test_edges")
    '''

    # g_train: new graph object with only non-hidden edges
    if directed == True:
        g_train = nx.DiGraph(adj_train)
    else:
        g_train = nx.Graph(adj_train)

    # Inspect train/test split
    if verbose >= 1:
        print ("Total nodes:", adj_sparse.shape[0])
        print ("Total edges:", int(adj_sparse.nnz/2) )# adj is symmetric, so nnz (num non-zero) = 2*num_edges
        print ("Training edges (positive):", len(train_edges))
        print ("Training edges (negative):", len(train_edges_false))
        print ("Validation edges (positive):", len(val_edges))
        print ("Validation edges (negative):", len(val_edges_false))
        print ("Test edges (positive):", len(test_edges))
        print ("Test edges (negative):", len(test_edges_false))
        print ("------------------------------------------------------")


    ### ---------- LINK PREDICTION BASELINES ---------- ###
    '''
    # Adamic-Adar

    aa_scores = adamic_adar_scores(g_train, train_test_split)
    lp_scores['aa'] = aa_scores
    if verbose >= 1:

        print ('Adamic-Adar Test ROC score: ', str(aa_scores['test_roc']))
        print ('Adamic-Adar Test AP score: ', str(aa_scores['test_ap']))
        print('Adamic-Adar Test ACC score: ', str(aa_scores['test_acc']))

    # Jaccard Coefficient
    jc_scores = jaccard_coefficient_scores(g_train, train_test_split)
    lp_scores['jc'] = jc_scores
    if verbose >= 1:

        print ('Jaccard Coefficient Test ROC score: ', str(jc_scores['test_roc']))
        print ('Jaccard Coefficient Test AP score: ', str(jc_scores['test_ap']))

    # Preferential Attachment
    pa_scores = preferential_attachment_scores(g_train, train_test_split)
    lp_scores['pa'] = pa_scores
    if verbose >= 1:

        print ('Preferential Attachment Test ROC score: ', str(pa_scores['test_roc']))
        print ('Preferential Attachment Test AP score: ', str(pa_scores['test_ap']))

    ### ---------- SPECTRAL CLUSTERING ---------- ###
    sc_scores = spectral_clustering_scores(train_test_split)
    lp_scores['sc'] = sc_scores
    if verbose >= 1:

        print ('Spectral Clustering Validation ROC score: ', str(sc_scores['val_roc']))
        print ('Spectral Clustering Validation AP score: ', str(sc_scores['val_ap']))
        print ('Spectral Clustering Test ROC score: ', str(sc_scores['test_roc']))
        print ('Spectral Clustering Test AP score: ', str(sc_scores['test_ap']))
    '''
    '''
    ## ---------- NODE2VEC ---------- ###
    # node2vec settings
    # NOTE: When p = q = 1, this is equivalent to DeepWalk
    P = 1 # Return hyperparameter
    Q = 1 # In-out hyperparameter
    WINDOW_SIZE = 10     # Context size for optimization
    NUM_WALKS = 10 # Number of walks per source
    WALK_LENGTH = 100 # Length of walk per source
    DIMENSIONS = 512 # Embedding dimension
    DIRECTED = False # Graph directed/undirected
    WORKERS = 8 # Num. parallel workers
    ITER = 1 # SGD epochs

    # Using bootstrapped edge embeddings + logistic regression
    n2v_edge_emb_scores = node2vec_scores(g_train, train_test_split,
        P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED, WORKERS, ITER,
        "edge-emb",
        verbose)
    lp_scores['n2v_edge_emb'] = n2v_edge_emb_scores

    if verbose >= 1:
 
        print ('node2vec (Edge Embeddings) Validation ROC score: ', str(n2v_edge_emb_scores['val_roc']))
        print ('node2vec (Edge Embeddings) Validation AP score: ', str(n2v_edge_emb_scores['val_ap']))
        print ('node2vec (Edge Embeddings) Test ROC score: ', str(n2v_edge_emb_scores['test_roc']))
        print ('node2vec (Edge Embeddings) Test AP score: ', str(n2v_edge_emb_scores['test_ap']))


    ## ---------- NODE2VEC Advanced ---------- ###
    # node2vec settings
    # NOTE: When p = q = 1, this is equivalent to DeepWalk
    P = 1 # Return hyperparameter
    Q = 8 # In-out hyperparameter
    WINDOW_SIZE = 10 # Context size for optimization
    NUM_WALKS = 10 # Number of walks per source
    WALK_LENGTH = 80 # Length of walk per source
    DIMENSIONS = 128 # Embedding dimension
    DIRECTED = False # Graph directed/undirected
    WORKERS = 8 # Num. parallel workers
    ITER = 1 # SGD epochs
    # deepwalk or improved algorithm.
    #0:deepwalk或node2vec
    #1:度最大
    #2：度前10%
    #3：按度大小选取概率等差排序
    #4：随机选取的概率与度成正比
    alg_type=4 

    # Using bootstrapped edge embeddings + logistic regression
    n2v_edge_emb_scores = node2vec_scores(g_train, train_test_split,
        P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED, WORKERS, ITER,
        "edge-emb",
        verbose,
        alg_type)
    if alg_type==0:        
        lp_scores['n2v_edge_emb'] = n2v_edge_emb_scores
    elif alg_type==1:
        lp_scores['n2v_edge_emb1'] = n2v_edge_emb_scores
    elif alg_type==2:
        lp_scores['n2v_edge_emb2,50%'] = n2v_edge_emb_scores
    elif alg_type==3:
        lp_scores['n2v_edge_emb3'] = n2v_edge_emb_scores
    else:
        lp_scores['n2v_edge_emb4'] = n2v_edge_emb_scores

    '''
    '''
    ## ---------- DEEPWALK ---------- ###
    deepwalk_scores = node2vec_scores(g_train, train_test_split,
        P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED, WORKERS, ITER,
        "edge-emb",
        verbose)
    lp_scores['deepwalk'] = deepwalk_scores

    if verbose >= 1:
 
        print ('deepwalk (Dot Product) Validation ROC score: ', str(deepwalk_scores['val_roc']))
        print ('deepwalk (Dot Product) Validation AP score: ', str(deepwalk_scores['val_ap']))
        print ('deepwalk (Dot Product) Test ROC score: ', str(deepwalk_scores['test_roc']))
        print ('deepwalk (Dot Product) Test AP score: ', str(deepwalk_scores['test_ap']))

    ###--------------------Line----------------------------------2019.0102###
    # Using bootstrapped edge embeddings + logistic regression
    
    line_calculate = linemethod(g_train, train_test_split)
    lp_scores['line'] = line_calculate

    if verbose >= 1:

        print ('line (Edge Embeddings) Test ROC score: ', str(line_calculate['test_roc']))
        print ('line (Edge Embeddings) Test AP score: ', str(line_calculate['test_ap']))

    ## ---------- GraphGAN ---------- ### 
    graphgan_scores=Graph_gan_score(g_train,train_test_split)
    lp_scores['GraphGAN']=graphgan_scores
    
    if verbose >= 1:

        print ('GraphGAN Test ROC score: ', str(graphgan_scores['test_roc']))
        print ('GraphGAN Test AP score: ', str(graphgan_scores['test_ap']))
    '''
    '''
    ## ---------- struc2vec ---------- ###     
    struc2vec_scores=Structovec_score(train_test_split)
    lp_scores['struc2vec']=struc2vec_scores
    
    if verbose >= 1:

        print ('struc2vec Validation ROC score: ', str(graphgan_scores['val_roc']))
        print ('struc2vec Validation AP score: ', str(graphgan_scores['val_ap']))
        print ('struc2vec Test ROC score: ', str(graphgan_scores['test_roc']))
        print ('struc2vec Test AP score: ', str(graphgan_scores['test_ap']))

    ##---------------------------maxdegree----------------##

    P = 1 # Return hyperparameter
    Q = 1 # In-out hyperparameter
    WINDOW_SIZE = 10 # Context size for optimization
    NUM_WALKS = 10 # Number of walks per source
    WALK_LENGTH = 80 # Length of walk per source
    DIMENSIONS = 128 # Embedding dimension
    DIRECTED = False # Graph directed/undirected
    WORKERS = 8 # Num. parallel workers
    ITER = 1 # SGD epochs

    # Using bootstrapped edge embeddings + logistic regression
    max_degree_scores = maxdegree_scores(g_train, train_test_split,
        P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED, WORKERS, ITER,
        "edge-emb",
        verbose)
    lp_scores['maxdegree_emb'] = max_degree_scores

    if verbose >= 1:
 
        print ('maxdegree (Edge Embeddings) Validation ROC score: ', str(max_degree_scores['val_roc']))
        print ('maxdegree (Edge Embeddings) Validation AP score: ', str(max_degree_scores['val_ap']))
        print ('maxdegree (Edge Embeddings) Test ROC score: ', str(max_degree_scores['test_roc']))
        print ('maxdegree (Edge Embeddings) Test AP score: ', str(max_degree_scores['test_ap']))
 
    ##------------------------topdegree-----------------------------------#
    P = 1 # Return hyperparameter
    Q = 1 # In-out hyperparameter
    WINDOW_SIZE = 10 # Context size for optimization
    NUM_WALKS = 10 # Number of walks per source
    WALK_LENGTH = 80 # Length of walk per source
    DIMENSIONS = 128 # Embedding dimension
    DIRECTED = False # Graph directed/undirected
    WORKERS = 8 # Num. parallel workers
    ITER = 1 # SGD epochs

    # Using bootstrapped edge embeddings + logistic regression
    top_degree_scores = topdegree_scores(g_train, train_test_split,
        P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED, WORKERS, ITER,
        "edge-emb",
        verbose)
    lp_scores['maxtop_emb'] = top_degree_scores

    if verbose >= 1:

        print ('topdegree (Edge Embeddings) Validation ROC score: ', str(top_degree_scores['val_roc']))
        print ('topdegree (Edge Embeddings) Validation AP score: ', str(top_degree_scores['val_ap']))
        print ('topdegree (Edge Embeddings) Test ROC score: ', str(top_degree_scores['test_roc']))
        print ('topdegree (Edge Embeddings) Test AP score: ', str(top_degree_scores['test_ap']))

    '''

    embedding_dim = 64
    walk_length = 40
    num_walks_node = 10
    constraint_learning_rate_scaling_factor = 0.1
    iterations = 1
    seed = 1
    window_size = 5
    num_layers = 1
    hidden_size = 64

    ##------------------------splitter-----------------------------------#

    edgelist_path = '/home/hxm/persona/data/edgelist' + g_name + '.edgelist'
    nx.write_edgelist(g_train, edgelist_path, data=False)

    # Using bootstrapped edge embeddings + logistic regression
    splitter_scores_res = splitter_scores(g_name, edgelist_path, train_test_split, embedding_dim=embedding_dim,
            walk_length=walk_length,
            num_walks_node=num_walks_node,
            constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
            iterations=iterations,
            seed=seed,
            window_size=window_size,
            local_clustering_fn=label_prop.label_propagation_communities)
    lp_scores['splitter'] = splitter_scores_res


    ##-------------------------------splitter+c----------------------#

    edgelist_path = '/home/hxm/persona/data/edgelist' + g_name + '.edgelist'
    nx.write_edgelist(g_train, edgelist_path, data=False)

    # Using bootstrapped edge embeddings + logistic regression
    splitter_scores_res = splitter_scores(g_name, edgelist_path, train_test_split, embedding_dim=embedding_dim,
            walk_length=walk_length,
            num_walks_node=num_walks_node,
            constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
            iterations=iterations,
            seed=seed,
            window_size=window_size,
            local_clustering_fn=components.connected_components)
    lp_scores['splitter-c'] = splitter_scores_res

    ##-------------------------------splitter+m----------------------#

    edgelist_path = '/home/hxm/persona/data/edgelist' + g_name + '.edgelist'
    nx.write_edgelist(g_train, edgelist_path, data=False)

    # Using bootstrapped edge embeddings + logistic regression
    splitter_scores_res = splitter_scores(g_name, edgelist_path, train_test_split, embedding_dim=embedding_dim,
            walk_length=walk_length,
            num_walks_node=num_walks_node,
            constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
            iterations=iterations,
            seed=seed,
            window_size=window_size,
            local_clustering_fn=modularity.greedy_modularity_communities)
    lp_scores['splitter-m'] = splitter_scores_res

    ##-------------------------------splitter+i----------------------#

    edgelist_path = '/home/hxm/persona/data/edgelist' + g_name + '.edgelist'
    nx.write_edgelist(g_train, edgelist_path, data=False)

    # Using bootstrapped edge embeddings + logistic regression
    splitter_scores_res = splitter_scores(g_name, edgelist_path, train_test_split, embedding_dim=embedding_dim,
            walk_length=walk_length,
            num_walks_node=num_walks_node,
            constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
            iterations=iterations,
            seed=seed,
            window_size=window_size,
            local_clustering_fn=ilfrs.ilfrs)
    lp_scores['splitter-i'] = splitter_scores_res

    ### ---------- RETURN RESULTS ---------- ###
    return lp_scores
