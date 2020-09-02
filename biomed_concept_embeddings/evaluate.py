from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json

from scipy.spatial.distance import cosine
from tqdm import  tqdm


def hls_palette(n_colors=6, h=.01, l=.6, s=.65):
    hues = np.linspace(0, 1, int(n_colors) + 1)[:-1]
    hues += h
    hues %= 1
    hues -= hues.astype(int)
    palette = [colorsys.hls_to_rgb(h_i, l if i % 2 == 0 else l + 0.2, s + 0.2 if i % 2 == 0 else s) for i, h_i in enumerate(hues)]
    return _ColorPalette(palette)


def pairwise_distances(embeddings):
    distances  = []
    for eid, e1 in enumerate(embeddings[:-1]):
        for e2 in embeddings[eid + 1:]:
            distances.append(cosine(e1, e2))
    return distances


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--embeddings')
    parser.add_argument('--vocab')
    parser.add_argument('--synsets')
    args = parser.parse_args()

    # loading
    with open(args.synsets, encoding='utf-8') as input_stream:
        synsets = [json.loads(line) for line in input_stream]
    synsets = [synset for synset in synsets if len(synset) > 15]
    print("Loaded synsets")
    concepts = [s for synset in synsets for s in synset]
    embeddings = np.load(args.embeddings)
    print("Loaded embeddings")
    vocab = pd.read_csv(args.vocab, names=['concept_id'], sep='\t', encoding='utf-8')
    print("Loaded vocab")

    # reducing dimension
    embeddings = [e for cui, e in zip(vocab.concept_id, embeddings) if cui in concepts]
    vocab = [cui for cui in vocab.concept_id if cui in concepts]
    print("Filtered vocab and embeddings")
    data = pd.DataFrame({'cui':  vocab, 'embedding': embeddings})
    distances = []
    for synset in tqdm(synsets):
        synset_embeddings = data[data.cui.isin(synset)].embedding.tolist()
        distances += pairwise_distances(synset_embeddings)
    print(len(distances))
    distances = np.array(distances)
    distances = distances[~np.isnan(distances)]
    print(distances.shape)
    print("Average distance between synonymic concepts is:",  np.mean(distances))
    print(distances.max())
