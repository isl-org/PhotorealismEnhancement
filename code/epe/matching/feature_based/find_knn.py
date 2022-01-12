from argparse import ArgumentParser
from pathlib import Path

import faiss
import numpy as np

if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument('file_src', type=Path, help="Path to feature file for source dataset.")
    p.add_argument('file_dst', type=Path, help="Path to feature file for target dataset.")
    p.add_argument('out', type=Path, help="Path to output file with matches.")
    p.add_argument('-k', type=int, help="Number of neighbours to sample. Default = 5.", default=5)
    args = p.parse_args()

    features_ref = np.load(args.file_src)['crops'].astype(np.float32)
    features_ref = features_ref / np.sqrt(np.sum(np.square(features_ref), axis=1, keepdims=True))

    dim = features_ref.shape[1]
    print(f'Found {features_ref.shape[0]} crops for source dataset.')

    features_nn = np.load(args.file_dst)['crops'].astype(np.float32)
    features_nn = features_nn / np.sqrt(np.sum(np.square(features_nn), axis=1, keepdims=True))
    assert features_nn.shape[1] == dim
    print(f'Found {features_nn.shape[0]} crops for target dataset.')

    nn_index = faiss.IndexFlatL2(dim)
    nn_index.add(features_nn)

    D,I = nn_index.search(features_ref, args.k)

    np.savez_compressed(f'{args.out}', ind=I, dist=D)
    pass
