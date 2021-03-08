import networkx as nx
import link_prediction6 as lp6
import link_prediction7 as lp7
import link_prediction8 as lp8
import link_prediction9 as lp9
import link_prediction10 as lp10
import pickle, json
import os

NUM_REPEATS = 1
RANDOM_SEED = 0
#FRAC_EDGES_HIDDEN = [0.1,0.25,0.5]
FRAC_EDGES_HIDDEN = [0.1] #划分比例，测试集占比

### ---------- Create Random NetworkX Graphs ---------- ###
# Dictionary to store all nx graphs
nx_graphs = {}
'''
# Small graphs
N_SMALL = 200
nx_graphs['er-small'] = nx.erdos_renyi_graph(n=N_SMALL, p=.02, seed=RANDOM_SEED) # Erdos-Renyi
nx_graphs['ws-small'] = nx.watts_strogatz_graph(n=N_SMALL, k=5, p=.1, seed=RANDOM_SEED) # Watts-Strogatz
nx_graphs['ba-small'] = nx.barabasi_albert_graph(n=N_SMALL, m=2, seed=RANDOM_SEED) # Barabasi-Albert
nx_graphs['pc-small'] = nx.powerlaw_cluster_graph(n=N_SMALL, m=2, p=.02, seed=RANDOM_SEED) # Powerlaw Cluster
nx_graphs['sbm-small'] = nx.random_partition_graph(sizes=[N_SMALL/10]*10, p_in=.1, p_out=.01, seed=RANDOM_SEED) # Stochastic Block Model
'''
#--------------------------------------------------------------------------------------

#nx_graphs['maayan-pdzbase']=nx.read_gpickle('data/avaliable/maayan-pdzbase.pck')
#nx_graphs['moreno_propro_propro']=nx.read_gpickle('data/avaliable/moreno_propro_propro.pck')
#nx_graphs['opsahl-powergrid']=nx.read_gpickle('data/avaliable/opsahl-powergrid.pck')
#nx_graphs['facebook_combined']=nx.read_gpickle('data/avaliable/facebook_combined.pck')
#nx_graphs['arenas-pgp']=nx.read_gpickle('data/avaliable/arenas-pgp.pck')
#nx_graphs['as-caida-2']=nx.read_gpickle('data/avaliable/as-caida20071105.pck')
#nx_graphs['ca-AstroPh']=nx.read_gpickle('data/avaliable/ca-AstroPh.pck')
#nx_graphs['arenas-email']=nx.read_gpickle('data/avaliable/arenas-email.pck')

#nx_graphs['subelj_cora_cora']=nx.read_edgelist('/home/hxm/persona/data/add/out.subelj_cora_cora')
#nx_graphs['CA-GrQc']=nx.read_edgelist('/home/hxm/persona/data/add/CA-GrQc.edgelist')
#nx_graphs['musae_git_edges']=nx.read_edgelist('/home/hxm/persona/data/avaliable/musae_git_edges.edgelist')
#nx_graphs['musae_facebook_edges']=nx.read_edgelist('/home/hxm/persona/data/avaliable/musae_facebook_edges.edgelist')


#nx_graphs['Email-Enron-2']=nx.read_edgelist('/home/hxm/persona/data/avaliable/Email-Enron.edgelist')
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

# 小网络
#nx_graphs['dolphin-baseline']=nx.read_gpickle('data/avaliable/dolphin.pck')
#nx_graphs['football']=nx.read_gpickle('data/avaliable/football.pck')
#nx_graphs['metabolic-1']=nx.read_gpickle('data/avaliable/metabolic.pck')
#nx_graphs['polbooks-2']=nx.read_gpickle('data/avaliable/polbooks.pck')
#nx_graphs['jazz']=nx.read_gpickle('data/avaliable/jazz.pck')
nx_graphs['ecoli']=nx.read_gpickle('data/avaliable/ecoli.pck')
#nx_graphs['health']=nx.read_gpickle('data/avaliable/health.pck')

#nx_graphs['usairport-2']=nx.read_gpickle('data/avaliable/usairport.pck')
#nx_graphs['dblpcite-2']=nx.read_gpickle('data/avaliable/dblpcite.pck')
'''
# Larger graphs
N_LARGE = 1000
nx_graphs['er-large'] = nx.erdos_renyi_graph(n=N_LARGE, p=.01, seed=RANDOM_SEED) # Erdos-Renyi
nx_graphs['ws-large'] = nx.watts_strogatz_graph(n=N_LARGE, k=3, p=.1, seed=RANDOM_SEED) # Watts-Strogatz
nx_graphs['ba-large'] = nx.barabasi_albert_graph(n=N_LARGE, m=2, seed=RANDOM_SEED) # Barabasi-Albert
nx_graphs['pc-large'] = nx.powerlaw_cluster_graph(n=N_LARGE, m=2, p=.02, seed=RANDOM_SEED) # Powerlaw Cluster
nx_graphs['sbm-large'] = nx.random_partition_graph(sizes=[N_LARGE/10]*10, p_in=.05, p_out=.005, seed=RANDOM_SEED) # Stochastic Block Model
'''
# Remove isolates from random graphs
for g_name, nx_g in nx_graphs.items():
    isolates = list(nx.isolates(nx_g))
    if len(isolates) > 0:
        for isolate_node in isolates:
            nx_graphs[g_name].remove_node(isolate_node)



### ---------- Run Link Prediction Tests ---------- ###
for i in range(NUM_REPEATS):
    ### ---------- NETWORKX ---------- ###
    nx_results = {}

    # Check existing experiment results, increment file number by 1
    past_results = os.listdir('./64_new_results/')   #输出
    '''
    experiment_num = 0
    experiment_file_name = 'nx-experiment-{}-results.json'.format(experiment_num)
    while (experiment_file_name in past_results):
        experiment_num += 1
        experiment_file_name = 'nx-experiment-{}-results.json'.format(experiment_num)
    
    NX_RESULTS_DIR = './new_results/' + experiment_file_name
    '''
    for g_name, nx_g in nx_graphs.items():
        NX_RESULTS_DIR = './64_new_results/' + g_name + '_snonn2.json'
     
    # Iterate over fractions of edges to hide
    for frac_hidden in FRAC_EDGES_HIDDEN:
        val_frac = 0.0
        test_frac = frac_hidden - val_frac
        
        # Iterate over each graph
        for g_name, nx_g in nx_graphs.items():
            # Remove the self-ring, and to undirected
            nx_g = nx_g.to_undirected()
            nx_g.remove_edges_from(nx_g.selfloop_edges())
            adj = nx.adjacency_matrix(nx_g)
            
            experiment_name = 'nx-{}-{}-hidden'.format(g_name, frac_hidden)
            print ("Current experiment: ", experiment_name)
            

            # lp6: Community-Network-Splitter
            # lp7: Community-Splitter(combined splitter and Network)
            # lp8: 同lp7, 并行用(lp7运行1-6，lp8运行1，7-11)
            # lp9: Community-Splitter(without nn)
            # lp10: Community-Splitter(single without nn)
            nx_results[experiment_name] = lp10.calculate_all_scores(g_name, adj, \
                                                         test_frac=test_frac, val_frac=val_frac, \
                                                         random_state=RANDOM_SEED, verbose=0)
            # Save experiment results at each iteration
            with open(NX_RESULTS_DIR, 'w') as f:
                json.dump(nx_results, f, indent=4)
            
    # Save final experiment results
    with open(NX_RESULTS_DIR, 'w') as f:
        json.dump(nx_results, f, indent=4)

