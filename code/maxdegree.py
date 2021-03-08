

# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import math


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def maxdegree_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        #alias_nodes = self.alias_nodes
        #alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                '''
                
                #只在度的前10%的邻居中随机选取一个
                nbrs_degrees={}
                for nbrs in cur_nbrs:
                    nbrs_degrees[nbrs]=G.degree(nbrs)#获得cur的邻居节点的度
                sort_degree=sorted(nbrs_degrees.items(),key=lambda nbrs_degrees:nbrs_degrees[1], reverse=True)
                index=int(math.ceil(len(cur_nbrs)/10))
                if len(walk) == 1:
                    if index<=1:
                        walk.append(sort_degree[int(index-1)][0])
                    else:
                        random_index=random.randint(0,index-1)
                        walk.append(sort_degree[random_index][0])
                else:
                    prev=walk[-2]
                    sort_degree=sorted(nbrs_degrees.items(),key=lambda nbrs_degrees:nbrs_degrees[1], reverse=True)
                    index=math.ceil(len(cur_nbrs)/10)
                    if prev==max(nbrs_degrees,key=nbrs_degrees.get) and index==1:
                        walk.append(sort_degree[1][0])
                    else:                       
                        if index<=1:
                            walk.append(sort_degree[int(index-1)][0])
                        else:
                            random_index=random.randint(0,index-1)
                            walk.append(sort_degree[random_index][0])
                        
                '''
                nbrs_degrees={}#不重复选之前选过的节点
                for nbrs in cur_nbrs:
                    nbrs_degrees[nbrs]=G.degree(nbrs)#获得cur的邻居节点的度
                if len(walk) == 1:
                    walk.append(max(nbrs_degrees,key=nbrs_degrees.get))
                else:
                    prev=walk[-2]
                    if prev==max(nbrs_degrees,key=nbrs_degrees.get) and len(cur_nbrs)>1:
                        sort_degree=sorted(nbrs_degrees.items(),key=lambda nbrs_degrees:nbrs_degrees[1], reverse=True)
                        walk.append(sort_degree[1][0])
                    else:
                        walk.append(max(nbrs_degrees,key=nbrs_degrees.get))
                
            else:
                break

        return walk
        

    def simulate_walks(self, num_walks, walk_length, verbose=True):
        '''
        Repeatedly simulate random walks from each node.
        num_walks: Number of walks per source
        walk_length:Length of walk per source
        
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        if verbose == True:
            print ('Walk iteration:')
        for walk_iter in range(num_walks):
            if verbose == True:
                print (str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)#将序列的所有元素随机排序
            for node in nodes:
                walks.append(self.maxdegree_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    从离散分布中计算非均匀采样的实用程序列表
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    利用混叠抽样从非均匀离散分布中抽取样本
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]