# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Implementation of the Ego-splitting Clustering Framework.
===============================
This is part of the implementation accompanying the WWW 2019 paper, [_Is a
Single Embedding Enough? Learning Node Representations that Capture Multiple
Social Contexts_](https://ai.google/research/pubs/pub46238).
The code in this file allows to create persona graphs, and to obtain overlapping
clusters using the persona graph method defined in the KDD 2017 paper
[_Ego-splitting Framework: from Non-Overlapping to Overlapping
Clusters_](http://epasto.org/papers/kdd2017.pdf).
Citing
------
If you find _Persona Embedding_ useful in your research, we ask that you cite
the following paper:
> Epasto, A., Perozzi, B., (2019).
> Is a Single Embedding Enough? Learning Node Representations that Capture
Multiple Social Contexts.
> In _The Web Conference_.
Example execution
------
python3 -m graph_embedding.persona.persona
  --input_graph=${graph} \
  --output_clustering=${clustering_output}
Where ${graph} is the path to a text file containing the graph and
${clustering_output} is the path to the output clustering.
The graph input format is a text file containing one edge per row represented
as its pair of node ids. The graph is supposed to be undirected.
For instance the file:
1 2
2 3
represents the triangle 1, 2, 3.
The output clustering format is a text file containing for each row one
(overlapping) cluster represented as the space-separted list of node ids in the
cluster.
The code uses two clustering algorithms local_clustering_method and
global_clustering_method respectively in each egonet and to split the persona ]
graph. The follow three options are allowed at the moment:
connected_components: the standard connected component algorithm.
label_prop: a label propagation based algorithm         社区划分
            (nx.label_prop.label_propagation_communities).
modularity: an algorithm optimizing modularity
            (nx.modularity.greedy_modularity_communities).
"""

import collections
import itertools
from absl import app
from absl import flags
from boltons.queueutils import HeapPriorityQueue
import networkx as nx
import networkx.algorithms.community.label_propagation as label_prop
import networkx.algorithms.community.modularity_max as modularity
import networkx.algorithms.components.connected as components
import LFM
import numpy as np
import community_ext
import ilfrs
import re

_CLUSTERING_FN = {
    'label_prop': label_prop.label_propagation_communities,
    'modularity': modularity.greedy_modularity_communities,
    'connected_components': components.connected_components,
    'ilfrs': ilfrs.ilfrs
}

flags.DEFINE_string(
    'input_graph', None,
    'The input graph path as a text file containing one edge per row, as the '
    'two node ids u v of the edge separated by a whitespace. The graph is '
    'assumed to be undirected. For example the file:\n1 2\n2 3\n represents '
    'the triangle 1 2 3')

flags.DEFINE_string(
    'output_clustering', None,
    'output path for the overallping clustering. The clustering is output as a '
    'text file where each row is a cluster, represented as the space-separated '
    'list of its node ids.')

flags.DEFINE_enum(
    'local_clustering_method', 'label_prop', _CLUSTERING_FN.keys(),
    'The method used for clustering the egonets of the graph. The options are '
    '"label_prop", "modularity" or "connected_components".')

flags.DEFINE_enum(
    'global_clustering_method', 'label_prop', _CLUSTERING_FN.keys(),
    'The method used for clustering the persona graph. The options are '
    'label_prop, modularity or connected_components.')

flags.DEFINE_integer('min_cluster_size', 5,
                     'Minimum size for an overlapping cluster to be output.')

flags.DEFINE_string(
    'output_persona_graph', None,
    'If set, it outputs the persona graph in the same format of the input graph'
    ' (text file).')
flags.DEFINE_string(
    'output_persona_graph_mapping', None,
    'If set, outputs the mapping of persona graphs ids to original graph '
    'ids as a text file where each row represents a node and it has two '
    'space-separated columns. The first column is the persona node id, while '
    'the second column is the original node id')

FLAGS = flags.FLAGS
def CreatePersonaGraph(graph,
                       g_name='',
                       clustering_fn=label_prop.label_propagation_communities,
                       persona_start_id=0):
  """The function creates the persona graph.
  Args:
    graph: Undirected graph represented as a dictionary of lists that maps each
      node id its list of neighbor ids;
    clustering_fn: A non-overlapping clustering algorithm function that takes in
      input a nx.Graph and outputs the a clustering. The output format is a list
      containing each partition as element. Each partition is in turn
      represented as a list of node ids. The default function is the networkx
      label_propagation_communities clustering algorithm.
    persona_start_id: The starting id (int) to use for the persona id
  Returns:
    A pair of (graph, mapping) where "graph" is an nx.Graph instance of the
    persona graph (which contains different nodes from the original graph) and
    "mapping" is a dict of the new node ids to the node ids in the original
    graph.The persona graph as nx.Graph, and the mapping of persona nodes to
    original node ids.
  """

  """
    'no_community_division': Indicates that there is no community division. 
    Creates a role graph that is isomorphic to the original graph and returns map 
    'LFM': Use the overlapping community discovery algorithm for the entire network
    'lcmi': Create a general map of three role maps using three kinds of community division
    'MIN': Choose the min community
  """

  if clustering_fn == 'lcmi':
      egonets = CreateEgonets(graph)
      node_neighbor_persona_id_map = collections.defaultdict(dict)
      persona_graph = nx.Graph()
      persona_to_original_mapping = dict()

      # Next id to allacate in persona graph.
      persona_id_counter = itertools.count(start=persona_start_id)
      node_num = 0
      for u, egonet in egonets.items():
          node_num += 1



      doc = open('/home/hxm/persona/64_splitter/' + g_name +'_label.txt', 'w+')
      print(node_num, file=doc)
      for u, egonet in egonets.items():
          partitioning = label_prop.label_propagation_communities(egonet)  # Clustering the egonet.
          seen_neighbors = set()  # Process each of the egonet's local clusters.
          part = []
          part_num = 0
          for partition in partitioning:
              persona_id = next(persona_id_counter)
              persona_to_original_mapping[persona_id] = u
              list_part = []
              part_num += 1
              for v in partition:
                  node_neighbor_persona_id_map[u][v] = persona_id
                  assert v not in seen_neighbors
                  seen_neighbors.add(v)
                  list_part.append(v)
              part.append(list_part.copy())
          print(u, part_num, file=doc)
          for list1 in part:
              for node in list1:
                  print(int(node), end=' ', file=doc)
              print(file=doc)
      doc.close()
      for u in graph.nodes():  # Process mapping to create persona graph.
          for v in graph.neighbors(u):
              if v == u:
                  continue
              assert v in node_neighbor_persona_id_map[u]
              u_p = node_neighbor_persona_id_map[u][v]
              assert u in node_neighbor_persona_id_map[v]
              v_p = node_neighbor_persona_id_map[v][u]
              persona_graph.add_edge(u_p, v_p)



      node_neighbor_persona_id_map2 = collections.defaultdict(dict)
      doc = open('/home/hxm/persona/64_splitter/' + g_name + '_compo.txt', 'w+')
      print(node_num, file=doc)
      for u, egonet in egonets.items():
          partitioning = components.connected_components(egonet)  # Clustering the egonet.
          seen_neighbors = set()  # Process each of the egonet's local clusters.
          part = []
          part_num = 0
          for partition in partitioning:
              persona_id = next(persona_id_counter)
              persona_to_original_mapping[persona_id] = u
              list_part = []
              part_num += 1
              for v in partition:
                  node_neighbor_persona_id_map2[u][v] = persona_id
                  assert v not in seen_neighbors
                  seen_neighbors.add(v)
                  list_part.append(v)
              part.append(list_part.copy())
          print(u, part_num, file=doc)
          for list1 in part:
              for node in list1:
                  print(int(node), end=' ', file=doc)
              print(file=doc)
      doc.close()
      for u in graph.nodes():  # Process mapping to create persona graph.
          for v in graph.neighbors(u):
              if v == u:
                  continue
              assert v in node_neighbor_persona_id_map2[u]
              u_p = node_neighbor_persona_id_map2[u][v]
              assert u in node_neighbor_persona_id_map2[v]
              v_p = node_neighbor_persona_id_map2[v][u]
              persona_graph.add_edge(u_p, v_p)


      node_neighbor_persona_id_map3 = collections.defaultdict(dict)
      doc = open('/home/hxm/persona/64_splitter/' + g_name + '_modul.txt', 'w+')
      print(node_num, file=doc)
      for u, egonet in egonets.items():
          partitioning = modularity.greedy_modularity_communities(egonet)  # Clustering the egonet.
          seen_neighbors = set()  # Process each of the egonet's local clusters.
          part = []
          part_num = 0
          for partition in partitioning:
              persona_id = next(persona_id_counter)
              persona_to_original_mapping[persona_id] = u
              list_part = []
              part_num += 1
              for v in partition:
                  node_neighbor_persona_id_map3[u][v] = persona_id
                  assert v not in seen_neighbors
                  seen_neighbors.add(v)
                  list_part.append(v)
              part.append(list_part.copy())
          print(u, part_num, file=doc)
          for list1 in part:
              for node in list1:
                  print(int(node), end=' ', file=doc)
              print(file=doc)
      doc.close()
      for u in graph.nodes():  # Process mapping to create persona graph.
          for v in graph.neighbors(u):
              if v == u:
                  continue
              assert v in node_neighbor_persona_id_map3[u]
              u_p = node_neighbor_persona_id_map3[u][v]
              assert u in node_neighbor_persona_id_map3[v]
              v_p = node_neighbor_persona_id_map3[v][u]
              persona_graph.add_edge(u_p, v_p)



      node_neighbor_persona_id_map4 = collections.defaultdict(dict)
      doc = open('/home/hxm/persona/64_splitter/' + g_name + '_ilfrs.txt', 'w+')
      print(node_num, file=doc)
      for u, egonet in egonets.items():
          partitioning = ilfrs.ilfrs(egonet)  # Clustering the egonet.
          seen_neighbors = set()  # Process each of the egonet's local clusters.
          part = []
          part_num = 0
          for partition in partitioning:
              persona_id = next(persona_id_counter)
              persona_to_original_mapping[persona_id] = u
              list_part = []
              part_num += 1
              for v in partition:
                  node_neighbor_persona_id_map4[u][v] = persona_id
                  assert v not in seen_neighbors
                  seen_neighbors.add(v)
                  list_part.append(v)
              part.append(list_part.copy())
          print(u, part_num, file=doc)
          for list1 in part:
              for node in list1:
                  print(int(node), end=' ', file=doc)
              print(file=doc)
      doc.close()
      for u in graph.nodes():  # Process mapping to create persona graph.
          for v in graph.neighbors(u):
              if v == u:
                  continue
              assert v in node_neighbor_persona_id_map4[u]
              u_p = node_neighbor_persona_id_map4[u][v]
              assert u in node_neighbor_persona_id_map4[v]
              v_p = node_neighbor_persona_id_map4[v][u]
              persona_graph.add_edge(u_p, v_p)

      return persona_graph, persona_to_original_mapping


  if clustering_fn == 'no_community_division':
    persona_graph = nx.Graph()
    persona_to_original_mapping = dict()
    for u in graph.nodes():
        for v in graph.neighbors(u):
            persona_graph.add_edge(str(int(u) + int(persona_start_id)), str(int(v) + int(persona_start_id)), weight=1)
            persona_to_original_mapping[int(u) + int(persona_start_id)] = u
            persona_to_original_mapping[int(v) + int(persona_start_id)] = v
    return persona_graph, persona_to_original_mapping

  if clustering_fn == 'LFM':
    graph2 = graph
    graph = nx.Graph()
    for u in graph2.nodes():
        for v in graph2.neighbors(u):
            if v == u:
                continue
            graph.add_edge(int(u), int(v))
    algorithm = LFM.LFM(graph, 0.3)
    communities = algorithm.execute()
    persona_graph = nx.Graph()
    persona_to_original_mapping = dict()
    g_train_matrix = np.array(nx.adjacency_matrix(graph).todense())
    print('The number of communities: ', len(communities))
    for c in communities:
        c = list(c)
        clen = len(c)
        cpre = persona_start_id
        for iindex in range(0, clen):
            persona_to_original_mapping[int(persona_start_id)] = str(c[iindex])
            persona_start_id = persona_start_id + 1
        for iindex in range(cpre, cpre + clen):
            for jindex in range(cpre, cpre + clen):
                if iindex == jindex:
                    continue
                if int(g_train_matrix[int(persona_to_original_mapping[iindex])][int(persona_to_original_mapping[jindex])]) == 1:
                    persona_graph.add_edge(str(iindex), str(jindex), weight=1)
    return persona_graph, persona_to_original_mapping



  if clustering_fn == 'MIN':
      egonets = CreateEgonets(graph)
      node_neighbor_persona_id_map = collections.defaultdict(dict)
      persona_graph = nx.Graph()
      persona_to_original_mapping = dict()

      # Next id to allacate in persona graph.
      persona_id_counter = itertools.count(start=persona_start_id)

      for u, egonet in egonets.items():
          partitioning = modularity.greedy_modularity_communities(egonet)  # Clustering the egonet.
          partitioning2 = label_prop.label_propagation_communities(egonet)
          partitioning3 = components.connected_components(egonet)
          partitioning4 = ilfrs.ilfrs(egonet)
          len1 = 0
          len2 = 0
          len3 = 0
          len4 = 0
          partitioning22 = []
          partitioning33 = []
          partitioning44 = []
          for partition in partitioning:
              len1 += 1
          for partition in partitioning2:
              len2 += 1
              listpart = []
              for v in partition:
                 listpart.append(v)
              partitioning22.append(frozenset(listpart))
          for partition in partitioning3:
              len3 += 1
              listpart = []
              for v in partition:
                 listpart.append(v)
              partitioning33.append(frozenset(listpart))
          for partition in partitioning4:
              len4 += 1
              listpart = []
              for v in partition:
                 listpart.append(v)
              partitioning44.append(frozenset(listpart))

          if len2 < len1:
              partitioning = partitioning22
              len1 = len2
          if len3 < len1:
              partitioning = partitioning33
              len1 = len3
          if len4 < len1:
              partitioning = partitioning44
              len1 = len4
          seen_neighbors = set()  # Process each of the egonet's local clusters.
          for partition in partitioning:
              persona_id = next(persona_id_counter)
              persona_to_original_mapping[persona_id] = u
              for v in partition:
                  node_neighbor_persona_id_map[u][v] = persona_id
                  assert v not in seen_neighbors
                  seen_neighbors.add(v)

      for u in graph.nodes():  # Process mapping to create persona graph.
          for v in graph.neighbors(u):
              if v == u:
                  continue
              assert v in node_neighbor_persona_id_map[u]
              u_p = node_neighbor_persona_id_map[u][v]
              assert u in node_neighbor_persona_id_map[v]
              v_p = node_neighbor_persona_id_map[v][u]
              persona_graph.add_edge(u_p, v_p)

      return persona_graph, persona_to_original_mapping

  if clustering_fn == 'MAX':
      egonets = CreateEgonets(graph)
      node_neighbor_persona_id_map = collections.defaultdict(dict)
      persona_graph = nx.Graph()
      persona_to_original_mapping = dict()

      # Next id to allacate in persona graph.
      persona_id_counter = itertools.count(start=persona_start_id)

      for u, egonet in egonets.items():
          partitioning = modularity.greedy_modularity_communities(egonet)  # Clustering the egonet.
          partitioning2 = label_prop.label_propagation_communities(egonet)
          partitioning3 = components.connected_components(egonet)
          partitioning4 = ilfrs.ilfrs(egonet)
          len1 = 0
          len2 = 0
          len3 = 0
          len4 = 0
          partitioning22 = []
          partitioning33 = []
          partitioning44 = []
          for partition in partitioning:
              len1 += 1
          for partition in partitioning2:
              len2 += 1
              listpart = []
              for v in partition:
                  listpart.append(v)
              partitioning22.append(frozenset(listpart))
          for partition in partitioning3:
              len3 += 1
              listpart = []
              for v in partition:
                  listpart.append(v)
              partitioning33.append(frozenset(listpart))
          for partition in partitioning4:
              len4 += 1
              listpart = []
              for v in partition:
                  listpart.append(v)
              partitioning44.append(frozenset(listpart))

          if len2 > len1:
              partitioning = partitioning22
              len1 = len2
          if len3 > len1:
              partitioning = partitioning33
              len1 = len3
          if len4 > len1:
              partitioning = partitioning44
              len1 = len4
          seen_neighbors = set()  # Process each of the egonet's local clusters.
          for partition in partitioning:
              persona_id = next(persona_id_counter)
              persona_to_original_mapping[persona_id] = u
              for v in partition:
                  node_neighbor_persona_id_map[u][v] = persona_id
                  assert v not in seen_neighbors
                  seen_neighbors.add(v)

      for u in graph.nodes():  # Process mapping to create persona graph.
          for v in graph.neighbors(u):
              if v == u:
                  continue
              assert v in node_neighbor_persona_id_map[u]
              u_p = node_neighbor_persona_id_map[u][v]
              assert u in node_neighbor_persona_id_map[v]
              v_p = node_neighbor_persona_id_map[v][u]
              persona_graph.add_edge(u_p, v_p)

      return persona_graph, persona_to_original_mapping



  if type(clustering_fn) == type('abc'):
      method = clustering_fn
      persona_graph = nx.Graph()
      persona_to_original_mapping = dict()
      persona_id_counter = itertools.count(start=persona_start_id)

      for c in method:
          if c == '-':  #不用重复计算lcmi，引入了-lcmi
              continue
          node_neighbor_persona_id_map = collections.defaultdict(dict)
          if c == 'l':
              doc = open('/home/hxm/persona/64_splitter/' + g_name +'_label.txt', 'r')
          elif c == 'c':
              doc = open('/home/hxm/persona/64_splitter/' + g_name + '_compo.txt', 'r')
          elif c == 'm':
              doc = open('/home/hxm/persona/64_splitter/' + g_name + '_modul.txt', 'r')
          elif c == 'i':
              doc = open('/home/hxm/persona/64_splitter/' + g_name + '_ilfrs.txt', 'r')
          lines = doc.readlines()
          res = re.findall(r'\d+', lines[0])
          res = list(map(int, res))
          node_num = res[0]
          j = 1
          for i in range(0, node_num):
              res = re.findall(r'\d+', lines[j])
              res = list(map(int, res))
              u = str(res[0])
              seen_neighbors = set()
              part_num = res[1]
              j += 1
              while part_num > 0:
                  res = re.findall(r'\d+', lines[j])
                  res = list(map(str, res))
                  persona_id = next(persona_id_counter)
                  persona_to_original_mapping[persona_id] = u
                  for v in res:
                      node_neighbor_persona_id_map[u][v] = persona_id
                      assert v not in seen_neighbors
                      seen_neighbors.add(v)
                  j += 1
                  part_num -= 1
          doc.close()
          for u in graph.nodes():
              for v in graph.neighbors(u):
                  if v == u:
                      continue
                  assert v in node_neighbor_persona_id_map[u]
                  u_p = node_neighbor_persona_id_map[u][v]
                  assert u in node_neighbor_persona_id_map[v]
                  v_p = node_neighbor_persona_id_map[v][u]
                  persona_graph.add_edge(u_p, v_p)

      return persona_graph, persona_to_original_mapping

  egonets = CreateEgonets(graph)
  node_neighbor_persona_id_map = collections.defaultdict(dict)
  persona_graph = nx.Graph()
  persona_to_original_mapping = dict()

  # Next id to allacate in persona graph.
  persona_id_counter = itertools.count(start=persona_start_id)

  for u, egonet in egonets.items():
    partitioning = clustering_fn(egonet)  # Clustering the egonet.
    seen_neighbors = set()  # Process each of the egonet's local clusters.
    for partition in partitioning:
      persona_id = next(persona_id_counter)
      persona_to_original_mapping[persona_id] = u
      for v in partition:
        node_neighbor_persona_id_map[u][v] = persona_id
        assert v not in seen_neighbors
        seen_neighbors.add(v)
  for u in graph.nodes():  # Process mapping to create persona graph.
    for v in graph.neighbors(u):
      if v == u:
        continue
      assert v in node_neighbor_persona_id_map[u]
      u_p = node_neighbor_persona_id_map[u][v]
      assert u in node_neighbor_persona_id_map[v]
      v_p = node_neighbor_persona_id_map[v][u]
      persona_graph.add_edge(u_p, v_p)

  return persona_graph, persona_to_original_mapping


def CreateEgonets(graph):
  """Given a graph, construct all the egonets of the graph.
  Args:
    graph: a nx.Graph instance for which the egonets have to be constructed.
  Returns:
    A dict mapping each node id to an instance of nx.Graph which represents the
    egonet for that node.
  """

  # This is used to not replicate the work for nodes that have been already
  # analyzed..
  completed_nodes = set()
  ego_egonet_map = collections.defaultdict(nx.Graph)

  # To reducing the running time the nodes are processed in increasing order of
  # degree.
  degrees_pq = HeapPriorityQueue()
  curr_degree = {}
  for node in graph.nodes:
    degrees_pq.add(node, -graph.degree[node])
    curr_degree[node] = graph.degree[node]

  # Ceating a set of the edges for fast membership testing.
  edge_set = set(graph.edges)

  while degrees_pq:
    node = degrees_pq.pop()
    # Update the priority queue decreasing the degree of the neighbor nodes.
    for neighbor in graph.neighbors(node):
      if neighbor == node:
        continue
      ego_egonet_map[node].add_node(
          neighbor)  # even if it is not part of a triangle it is there.
      # We decrease the degree of the nodes still not processed.
      if neighbor not in completed_nodes:
        curr_degree[neighbor] -= 1
        degrees_pq.remove(neighbor)
        degrees_pq.add(neighbor, -curr_degree[neighbor])

    # Construct egonet of u by enumerating all triangles to which u belong
    # because each edge in a triangle is an edge in the egonets of the triangle
    # vertices  and vice versa.
    not_removed = []
    for neighbor in graph.neighbors(node):
      if neighbor not in completed_nodes:
        not_removed.append(neighbor)
    for pos_u, u in enumerate(not_removed):
      for v  in not_removed[pos_u+1:]:
        if (u, v) in edge_set or (v, u) in edge_set:
          ego_egonet_map[node].add_edge(u, v)
          ego_egonet_map[u].add_edge(node, v)
          ego_egonet_map[v].add_edge(u, node)

    completed_nodes.add(node)
  return ego_egonet_map


def PersonaOverlappingClustering(graph, local_clustering_fn,
                                 global_clustering_fn, min_component_size):
  """Computes an overlapping clustering of graph using the Ego-Splitting method.
  Args:
    graph: a networkx graph for which the egonets have to be constructed.
    local_clustering_fn: method used for clustering the egonets.
    global_clustering_fn: method used for clustering the persona graph.
    min_component_size: minimum size of a cluster to be output.
  Returns:
    The a overlapping clustering (list of sets of node ids), the persona graph
    (nx.Graph) and the persona node
    id mapping (dictionary of int to string) .
  """

  persona_graph, persona_id_mapping = CreatePersonaGraph(
      graph, local_clustering_fn)
  non_overlapping_clustering = global_clustering_fn(persona_graph)
  overlapping_clustering = set()
  for cluster in non_overlapping_clustering:
    if len(cluster) < min_component_size:
      continue
    cluster_original_graph = set([persona_id_mapping[c] for c in cluster])
    cluster_original_graph = list(cluster_original_graph)
    cluster_original_graph.sort()
    overlapping_clustering.add(tuple(cluster_original_graph))
  return list(overlapping_clustering), persona_graph, persona_id_mapping


def main(argv=()):
  del argv  # Unused.
  graph = nx.read_edgelist(FLAGS.input_graph, create_using=nx.Graph)

  local_clustering_fn = _CLUSTERING_FN[FLAGS.local_clustering_method]
  global_clustering_fn = _CLUSTERING_FN[FLAGS.global_clustering_method]

  clustering, persona_graph, persona_id_mapping = PersonaOverlappingClustering(
      graph, local_clustering_fn, global_clustering_fn, FLAGS.min_cluster_size)

  with open(FLAGS.output_clustering, 'w') as outfile:
    for cluster in clustering:
      outfile.write(' '.join([str(x) for x in cluster]) + '\n')

  if FLAGS.output_persona_graph is not None:
    nx.write_edgelist(persona_graph, FLAGS.output_persona_graph)
  if FLAGS.output_persona_graph_mapping is not None:
    with open(FLAGS.output_persona_graph_mapping, 'w') as outfile:
      for persona_node, original_node in persona_id_mapping.items():
        outfile.write('{} {}\n'.format(persona_node, original_node))
  return 0


if __name__ == '__main__':
  flags.mark_flag_as_required('input_graph')
  flags.mark_flag_as_required('output_clustering')
  app.run(main)