import networkx as nx
import community_ext


def ilfrs(G):
    methods = {'ilfrs'}
    m = sum([d.get('weight', 1) for u, v, d in G.edges(data=True)])
    if m == 0:
        label_for_node = dict((i, v) for i, v in enumerate(G.nodes()))
        return [
            frozenset([label_for_node[i] for i in label_for_node])
        ]

    for method in methods:
        # a starting parameter value depends on the method
        work_par = 0.5


        # now start the iterative process
        prev_par, it = -1, 0
        prev_pars = set()
        while abs(work_par-prev_par)>1e-5: # stop if the size of improvement too small
            it += 1
            if it>100: break # stop after 100th iteration

            # update the parameter value
            prev_par = work_par
            if prev_par in prev_pars: break # stop if we are in the cycle
            prev_pars.add(prev_par)

            # find the optimal partition with the current parameter value
            partition = community_ext.best_partition(G,model=method,pars={'mu':work_par})


            # calculate optimal parameter value for the current partition
            work_par = community_ext.estimate_mu(G,partition)

        # calculate and print the scores of resulting partition
        partition_num = 0
        for part in partition:
            if partition[part] > partition_num:
                partition_num = partition[part]
        partition_num += 1
        t = 0
        partitioning = []
        while t < partition_num:
            listpart = []
            for part in partition:
                if partition[part] == t:
                    listpart.append(part)
            partitioning.append(frozenset(listpart.copy()))
            t += 1
    return partitioning