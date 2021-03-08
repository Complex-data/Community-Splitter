import networkx as nx
import numpy as np
import networkx.algorithms.community.label_propagation as label_prop
import networkx.algorithms.community.modularity_max as modularity
import networkx.algorithms.components.connected as components
import ilfrs
import community_ext

nx_graphs = {}
#nx_graphs['maayan-pdzbase']=nx.read_gpickle('data/avaliable/maayan-pdzbase.pck')
#nx_graphs['moreno_propro_propro']=nx.read_gpickle('data/avaliable/moreno_propro_propro.pck')
#nx_graphs['opsahl-powergrid']=nx.read_gpickle('data/avaliable/opsahl-powergrid.pck')
#nx_graphs['facebook_combined']=nx.read_gpickle('data/avaliable/facebook_combined.pck')
#nx_graphs['arenas-pgp']=nx.read_gpickle('data/avaliable/arenas-pgp.pck')
#nx_graphs['as-caida']=nx.read_gpickle('data/avaliable/as-caida20071105.pck')
#nx_graphs['ca-AstroPh']=nx.read_gpickle('data/avaliable/ca-AstroPh.pck')
#nx_graphs['arenas-email']=nx.read_gpickle('data/avaliable/arenas-email.pck')

#nx_graphs['subelj_cora_cora']=nx.read_edgelist('/home/hxm/persona/data/add/out.subelj_cora_cora')
#nx_graphs['CA-GrQc']=nx.read_edgelist('/home/hxm/persona/data/add/CA-GrQc.edgelist')
#nx_graphs['musae_git_edges']=nx.read_edgelist('/home/hxm/persona/data/avaliable/musae_git_edges.edgelist')
#nx_graphs['musae_facebook_edges']=nx.read_edgelist('/home/hxm/persona/data/avaliable/musae_facebook_edges.edgelist')


#nx_graphs['Email-Enron']=nx.read_edgelist('/home/hxm/persona/data/avaliable/Email-Enron.edgelist')
#nx_graphs['deezer_europe']=nx.read_edgelist('/home/hxm/persona/data/avaliable/deezer_europe.edgelist')
#nx_graphs['musae_wiki']=nx.read_edgelist('/home/hxm/persona/data/avaliable/musae_wiki.edgelist')
#nx_graphs['p2p-Gnutella30']=nx.read_edgelist('/home/hxm/persona/data/avaliable/p2p-Gnutella30.txt')
#nx_graphs['brazil-airports']=nx.read_gpickle('data/avaliable/brazil-airports.pck')
#nx_graphs['ego-facebook']=nx.read_gpickle('data/avaliable/ego-facebook.pck')
#nx_graphs['petster-friendships-hamster-uniq']=nx.read_gpickle('data/avaliable/petster-friendships-hamster-uniq.pck')
#nx_graphs['petster-hamster']=nx.read_gpickle('data/avaliable/petster-hamster.pck')
#nx_graphs['CA-GrQc']=nx.read_gpickle('data/avaliable/CA-GrQc.pck')   #id is not from 1 to n
#nx_graphs['ppi']=nx.read_gpickle('data/avaliable/ppi.pck') #id is not from 1 to n
#nx_graphs['power-modul']=nx.read_gpickle('data/avaliable/power.pck')
#nx_graphs['reactome']=nx.read_gpickle('data/avaliable/reactome.pck')#id is not from 1 to n
#nx_graphs['dblp']=nx.read_gpickle('data/avaliable/dblpGEQ3.pck')  #too large
#nx_graphs['tntp-ChicagoRegional']=nx.read_gpickle('data/avaliable/tntp-ChicagoRegional.pck') #can't generate the test edges(0)

#nx_graphs['dolphin']=nx.read_gpickle('data/avaliable/dolphin.pck')
#nx_graphs['football']=nx.read_gpickle('data/avaliable/football.pck')
#nx_graphs['metabolic']=nx.read_gpickle('data/avaliable/metabolic.pck')
#nx_graphs['polbooks']=nx.read_gpickle('data/avaliable/polbooks.pck')
#nx_graphs['jazz']=nx.read_gpickle('data/avaliable/jazz.pck')
#nx_graphs['usairport']=nx.read_gpickle('data/avaliable/usairport.pck')
#nx_graphs['dblpcite']=nx.read_gpickle('data/avaliable/dblpcite.pck')
#nx_graphs['ecoli']=nx.read_gpickle('data/avaliable/ecoli.pck')
nx_graphs['health']=nx.read_gpickle('data/avaliable/health.pck')

for g_name, nx_g in nx_graphs.items():
    isolates = list(nx.isolates(nx_g))
    if len(isolates) > 0:
        for isolate_node in isolates:
            nx_graphs[g_name].remove_node(isolate_node)

nx_g = nx_g.to_undirected()
nx_g.remove_edges_from(nx_g.selfloop_edges())
adj = nx.adjacency_matrix(nx_g)
G = nx.Graph(adj)

partition1 = label_prop.label_propagation_communities(G)
partition2 = components.connected_components(G)
partition3 = modularity.greedy_modularity_communities(G)
partition4 = ilfrs.ilfrs(G)
dict1 = {}
dict2 = {}
dict3 = {}
dict4 = {}
c1 = 0
c2 = 0
c3 = 0
c4 = 0

dict = {}
for part in partition1:
    for v in part:
        dict[int(v)] = int(c1)
    c1 += 1

for key in sorted(dict):
    dict1[int(key)] = dict[key]

dict = {}
for part in partition2:
    for v in part:
        dict[int(v)] = int(c2)
    c2 += 1

for key in sorted(dict):
    dict2[int(key)] = dict[key]

dict = {}
for part in partition3:
    for v in part:
        dict[int(v)] = int(c3)
    c3 += 1

for key in sorted(dict):
    dict3[int(key)] = dict[key]

dict = {}
for part in partition4:
    for v in part:
        dict[int(v)] = int(c4)
    c4 += 1

for key in sorted(dict):
    dict4[int(key)] = dict[key]

print(g_name)

res = community_ext.compare_partitions(dict1, dict2)
print('l-c: ', res['nmi'], res['jaccard'])

res = community_ext.compare_partitions(dict1, dict3)
print('l-m: ', res['nmi'], res['jaccard'])

res = community_ext.compare_partitions(dict1, dict4)
print('l-i: ', res['nmi'], res['jaccard'])

res = community_ext.compare_partitions(dict2, dict3)
print('c-m: ', res['nmi'], res['jaccard'])

res = community_ext.compare_partitions(dict2, dict4)
print('c-i: ', res['nmi'], res['jaccard'])

res = community_ext.compare_partitions(dict3, dict4)
print('m-i: ', res['nmi'], res['jaccard'])

'''
# 画图用
graph1 = nx.Graph()
list1 = []
graph2 = nx.Graph()
list2 = []
for key in dict1:
    if dict1[key] == dict1[0]:
        list1.append(key)
for key in dict2:
    if dict2[key] == dict2[0]:
        list2.append(key)

adj = adj.todense()
adj = np.array(adj)


for i in list1:
    for j in list1:
        if i != j:
            if adj[i][j] == 1:
                graph1.add_edge(i, j)
for i in list2:
    for j in list2:
        if i != j:
            if adj[i][j] == 1:
                graph2.add_edge(i, j)
'''
