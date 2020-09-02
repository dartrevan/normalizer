import colorsys
from argparse import ArgumentParser 
from sklearn.manifold import TSNE 
import numpy as np 
import pandas as pd 
import json 
from sklearn.ensemble import IsolationForest

import seaborn as sns
from seaborn.palettes import _color_to_rgb, _ColorPalette
import matplotlib.pyplot as plt


def hls_palette(n_colors=6, h=.01, l=.6, s=.65):
    hues = np.linspace(0, 1, int(n_colors) + 1)[:-1]
    hues += h
    hues %= 1
    hues -= hues.astype(int)
    palette = [colorsys.hls_to_rgb(h_i, l if i % 2 == 0 else l + 0.2, s + 0.2 if i % 2 == 0 else s) for i, h_i in enumerate(hues)]
    return _ColorPalette(palette)


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
    ld_embeddings = TSNE(n_components=2).fit_transform(embeddings)
    ld_embeddings = {cui: e for cui, e in zip(vocab, ld_embeddings)}
    print("Dim reduction")

    # preparing data for visualization
    vis_data = pd.DataFrame({'cui': synsets})
    vis_data['synset_id'] = list(range(vis_data.shape[0]))
    vis_data = vis_data.explode('cui')
    vis_data = vis_data.dropna()
    vis_data['embedding'] = vis_data.cui.apply(ld_embeddings.get)
    vis_data = vis_data.dropna()
    #print(vis_data['embedding'].head())
    vis_data['x'] = vis_data.embedding.apply(lambda t: t[0])
    vis_data['y'] = vis_data.embedding.apply(lambda t: t[1])
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1), max_features=2)
    print(vis_data[['x', 'y']].values.shape)
    model.fit(vis_data[['x', 'y']].values)
    vis_data['anomaly'] = model.predict(vis_data[['x', 'y']].values)
    vis_data = vis_data[vis_data.anomaly==1]
    # plotting
    syns_count = vis_data.synset_id.drop_duplicates().shape[0]
    markers = ['o', 'v', 's'] * 100
    markers = markers[:syns_count]
    sns_plot = sns.lmplot(x='x', y='y', data=vis_data, hue='synset_id',
               fit_reg=False, legend=True, legend_out=True, palette=hls_palette(syns_count, l=.4, s=0.79), markers=markers)
    sns_plot.savefig("output.png")
