'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import node2vec as n2v


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist', help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb', help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true', default=False,
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--directed', dest='directed', action='store_true', default=False,
                        help='Graph is (un)directed. Default is undirected.')

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    model.wv.save_word2vec_format(args.output)

    return


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = n2v.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks)


def convert_emd_to_npy_format(file_in, file_out):
    def str_to_num(str_temmp):
        return float(str_temmp)

    def by_node_id(n):
        return n[0]

    def remove_node_id(n):
        return n[1:]

    all_nodes = list()
    with open(file_in, 'r') as f:
        for line in f:
            line = line.split(' ')
            line = list(map(str_to_num, line))
            all_nodes.append(tuple(line))

    all_nodes = all_nodes[1:]  # remove the first line: num_of_nodes, dim_len
    all_nodes_sorted = sorted(all_nodes, key=by_node_id)
    all_nodes_remove_id = list(map(remove_node_id, all_nodes_sorted))
    node_array = np.array(all_nodes_remove_id)
    np.save(file_out, node_array)

    print("done.")


args = parse_args()
main(args)

# --input ../dataset_cmu/edge/edge_pair.ungraph  --output ../dataset_cmu/edge/out_of_order.emd  --dimensions 128
convert_emd_to_npy_format(file_in='../dataset_cmu/edge/out_of_order.emd', file_out="../dataset_cmu/node2vec_dim128")
