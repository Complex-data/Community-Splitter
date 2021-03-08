import os
import collections
import tqdm
import multiprocessing
import numpy as np
import tensorflow as tf
import generator
import discriminator
import graphutils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GraphGAN(object):
    def __init__(self,g_train,n_node):
        print("reading graphs...")
        #self.n_node, self.graph = graphutils.read_edges(config.train_filename, config.test_filename)
        #self.n_node=graphutils.read_edges('/home/hxm/link_prediciton/data/net/out.moreno_propro_propro')
        self.n_node=n_node
        self.graph=g_train
        
        self.batch_size_gen = 64  # batch size for the generator
        self.batch_size_dis = 64  # batch size for the discriminator
        self.lambda_gen = 1e-5  # l2 loss regulation weight for the generator
        self.lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator
        self.n_sample_gen = 20  # number of samples for the generator
        self.lr_gen = 1e-3  # learning rate for the generator
        self.lr_dis = 1e-3  # learning rate for the discriminator
        self.n_epochs = 2  # number of outer loops
        self.n_epochs_gen = 1  # number of inner loops for the generator
        self.n_epochs_dis = 1  # number of inner loops for the discriminator
        self.gen_interval = self.n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
        self.dis_interval = self.n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations
        self.update_ratio = 0.5    # updating ratio when choose the trees
        
        # model saving
        self.load_model = False  # whether loading existing model for initialization
        self.save_steps = 10
        
        # other hyper-parameters
        self.n_emb = 20
        self.multi_processing = False  # whether using multi-processing to construct BFS-trees
        self.window_size = 2
        
        self.model_log = "./log/"
    
        self.root_nodes = [i for i in range(self.n_node)]
        

        #Randomly initialize node embedding
        
        self.node_embed_init_d = np.random.rand(n_node, self.n_emb)
        self.node_embed_init_g = np.random.rand(n_node, self.n_emb)
        
        #self.node_embed_init_d = graphutils.read_embeddings(filename='/home/hxm/link_prediciton/emb/maayan-pdzbase.emb',
        #                                               n_node=self.n_node,
        #                                               n_embed=self.n_emb)
        #self.node_embed_init_g = graphutils.read_embeddings(filename='/home/hxm/link_prediciton/emb/maayan-pdzbase.emb',
        #                                               n_node=self.n_node,
        #                                               n_embed=self.n_emb)
        
        # construct or read BFS-trees
        self.trees = None
        if self.trees==None:
            print("constructing BFS-trees...")
            if self.multi_processing:
                self.construct_trees_with_mp(self.root_nodes)
            else:
                self.trees = self.construct_trees(self.root_nodes)

        print("building GAN model...")
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()

        self.latest_checkpoint = tf.train.latest_checkpoint(self.model_log)
        self.saver = tf.train.Saver()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)
        
    def construct_trees_with_mp(self, nodes):
        """use the multiprocessing to speed up trees construction
        Args:
            nodes: the list of nodes in the graph
        """

        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)
        new_nodes = []
        n_node_per_core = self.n_node // cores
        for i in range(cores):
            if i != cores - 1:
                new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
            else:
                new_nodes.append(nodes[i * n_node_per_core:])
        self.trees = {}
        trees_result = pool.map(self.construct_trees, new_nodes)
        for tree in trees_result:
            self.trees.update(tree)

    def construct_trees(self, nodes):
        """use BFS algorithm to construct the BFS-trees
        Args:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees

    def build_generator(self):
        """initializing the generator"""

        with tf.variable_scope("generator"):
            self.generator = generator.Generator(n_node=self.n_node, node_emd_init=self.node_embed_init_g)

    def build_discriminator(self):
        """initializing the discriminator"""

        with tf.variable_scope("discriminator"):
            self.discriminator = discriminator.Discriminator(n_node=self.n_node, node_emd_init=self.node_embed_init_d)

    def train(self):
        # restore the model from the latest checkpoint if exists
        checkpoint = tf.train.get_checkpoint_state(self.model_log)
        if checkpoint and checkpoint.model_checkpoint_path and self.load_model:
            print("loading the checkpoint: %s" % checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)


        print("start training...")
        for epoch in range(self.n_epochs):
            print("epoch %d" % epoch)

            # save the model
            if epoch > 0 and epoch % self.save_steps == 0:
                self.saver.save(self.sess, self.model_log + "model.checkpoint")

            # D-steps
            center_nodes = []
            neighbor_nodes = []
            labels = []
            for d_epoch in range(self.n_epochs_dis):
                # generate new nodes for the discriminator for every dis_interval iterations
                if d_epoch % self.dis_interval == 0:
                    center_nodes, neighbor_nodes, labels = self.prepare_data_for_d()
                # training
                train_size = len(center_nodes)
                start_list = list(range(0, train_size, self.batch_size_dis))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + self.batch_size_dis
                    self.sess.run(self.discriminator.d_updates,
                                  feed_dict={self.discriminator.node_id: np.array(center_nodes[start:end]),
                                             self.discriminator.node_neighbor_id: np.array(neighbor_nodes[start:end]),
                                             self.discriminator.label: np.array(labels[start:end])})

            # G-steps
            node_1 = []
            node_2 = []
            reward = []
            for g_epoch in range(self.n_epochs_gen):
                if g_epoch % self.gen_interval == 0:
                    node_1, node_2, reward = self.prepare_data_for_g()

                # training
                train_size = len(node_1)
                start_list = list(range(0, train_size, self.batch_size_gen))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + self.batch_size_gen
                    self.sess.run(self.generator.g_updates,
                                  feed_dict={self.generator.node_id: np.array(node_1[start:end]),
                                             self.generator.node_neighbor_id: np.array(node_2[start:end]),
                                             self.generator.reward: np.array(reward[start:end])})
        
        modes = [self.generator, self.discriminator]
        embedding_matrix = self.sess.run(modes[1].embedding_matrix)
        return embedding_matrix

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in self.root_nodes:
            if np.random.rand() < self.update_ratio:
                pos = self.graph[i]
                neg, _ = self.sample(i, self.trees[i], len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    # negative samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        return center_nodes, neighbor_nodes, labels

    def prepare_data_for_g(self):
        """sample nodes for the generator"""
        paths = []
        for i in self.root_nodes:
            if np.random.rand() < self.update_ratio:
                sample, paths_from_i = self.sample(i, self.trees[i], self.n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return node_1, node_2, reward

    def sample(self, root, tree, sample_num, for_d):
        """ sample nodes from BFS-tree
        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        all_score = self.sess.run(self.generator.all_score)
        samples = []
        paths = []
        n = 0

        def softmax(x):
            e_x = np.exp(x - np.max(x))  # for computation stability
            return e_x / e_x.sum()

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive samples)
                    if node_neighbor == [root]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_probability = all_score[current_node, node_neighbor]
                relevance_probability = softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    def get_node_pairs_from_path(self,path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """

        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - self.window_size, 0), min(i + self.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs

