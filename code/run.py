import networkx as nx
import link_prediction as lp
import pickle, json
import os

NUM_REPEATS = 1
RANDOM_SEED = 0
#FRAC_EDGES_HIDDEN = [0.1, 0.25, 0.5]
FRAC_EDGES_HIDDEN = [0.1]

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
#nx_graphs['as-caida20071105']=nx.read_gpickle('data/avaliable/as-caida20071105.pck')
#nx_graphs['maayan-vidal']=nx.read_gpickle('data/avaliable/maayan-vidal.pck')
#nx_graphs['arenas-email']=nx.read_gpickle('data/avaliable/arenas-email.pck')
nx_graphs['karate']=nx.read_edgelist('/home/hxm/persona/karate.edgelist')
#nx_graphs['wiki_vote']=nx.read_edgelist('/home/hxm/persona/Wiki-Vote.txt')
#nx_graphs['brazil-airports']=nx.read_gpickle('data/avaliable/brazil-airports.pck')
#nx_graphs['ego-facebook']=nx.read_gpickle('data/avaliable/ego-facebook.pck')
#nx_graphs['petster-friendships-hamster-uniq']=nx.read_gpickle('data/avaliable/petster-friendships-hamster-uniq.pck')
#nx_graphs['petster-hamster']=nx.read_gpickle('data/avaliable/petster-hamster.pck')
#nx_graphs['ca-AstroPh']=nx.read_gpickle('data/avaliable/ca-AstroPh.pck')

#nx_graphs['CA-GrQc']=nx.read_gpickle('data/avaliable/CA-GrQc.pck')   #id is not from 1 to n
#nx_graphs['CA-HepTh']=nx.read_gpickle('data/avaliable/CA-HepTh.pck') #id is not from 1 to n
#nx_graphs['reactome']=nx.read_gpickle('data/avaliable/reactome.pck')#id is not from 1 to n
#nx_graphs['dblp']=nx.read_gpickle('data/avaliable/dblpGEQ3.pck')  #too large
#nx_graphs['tntp-ChicagoRegional']=nx.read_gpickle('data/avaliable/tntp-ChicagoRegional.pck') #can't generate the test edges(0)
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
    past_results = os.listdir('./results/')
   
    experiment_num = 0
    experiment_file_name = 'nx-experiment-{}-results.json'.format(experiment_num)
    while (experiment_file_name in past_results):
        experiment_num += 1
        experiment_file_name = 'nx-experiment-{}-results.json'.format(experiment_num)
    
    NX_RESULTS_DIR = './results/' + experiment_file_name
    

    # Iterate over fractions of edges to hide
    for frac_hidden in FRAC_EDGES_HIDDEN:
        val_frac = 0.0
        test_frac = frac_hidden - val_frac
        
        # Iterate over each graph
        for g_name, nx_g in nx_graphs.items():
            adj = nx.adjacency_matrix(nx_g)
            #print("this is the adj in run.py", adj)
            experiment_name = 'nx-{}-{}-hidden'.format(g_name, frac_hidden)
            print("Current experiment:" , experiment_name)
            
            # Run all link prediction methods on current graph, store results
            nx_results[experiment_name] = lp.calculate_all_scores(adj, \
                                                         test_frac=test_frac, val_frac=val_frac, \
                                                         random_state=RANDOM_SEED, verbose=0)

            # Save experiment results at each iteration
            with open(NX_RESULTS_DIR, 'w') as f:
                json.dump(nx_results, f, indent=4)
            
    # Save final experiment results
    with open(NX_RESULTS_DIR, 'w') as f:
        json.dump(nx_results, f, indent=4)
