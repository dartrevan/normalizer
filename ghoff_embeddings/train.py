import json
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from gensim.models import KeyedVectors
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
from enum import Enum
from argparse import ArgumentParser
import pandas as pd
from nltk.tokenize import word_tokenize


class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class Embedder(nn.Module):

    @staticmethod
    def create_emb_layer(weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.shape
        weights_matrix = torch.Tensor(weights_matrix)
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer

    def __init__(self, word2index, weights_matrix=None, input_size=None, hidden_size=None):
        super(Embedder, self).__init__()
        self.word2index = word2index
        if weights_matrix is not None:
            self.embedding = self.create_emb_layer(weights_matrix)
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)

    def forward(self, input):
        return self.embedding(input)

    def convert2idx(self, token):
        return self.word2index[token]


def load_embeddings(path):
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    word2index = {}
    embeddings = []
    for token in model.wv.vocab.keys():
        word2index[token] = len(word2index)
        embeddings.append(model[token])
    embeddings = np.vstack(embeddings)
    return word2index, embeddings


def sample_pos_neg_words(entity_id, token_entity_distribution, token_distribution):
    pos_word_id = np.random.choice(token_distribution.shape[0], 1, p=token_entity_distribution[entity_id])
    neg_word_id = np.random.choice(token_distribution.shape[0], 1, p=token_distribution)
    return pos_word_id, neg_word_id



def get_token_entity_distribution(entities, labels, word2index, label2index):
    d = np.zeros([len(label2index), len(word2index)])
    for entity, label in zip(entities, labels):
        #print(label)
        if label not in label2index: continue
        label_idx = label2index[label]
        if not isinstance(entity, str): continue
        for token in word_tokenize(entity):
            #print(token)
            #if not isinstance(token, str): continue
            if token not in word2index: continue
            token_idx = word2index[token]
            d[label_idx, token_idx] += 1
    return d + 0.1


def get_token_distribution(entities, word2index):
    d = np.zeros(len(word2index))
    for entity in entities:
        if not isinstance(entity, str): continue
        for token in word_tokenize(entity):
            #if not isinstance(token, str): continue
            if token not in word2index: continue
            token_idx = word2index[token]
            d[token_idx] += 1
    return d + 0.7


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--entities')
    parser.add_argument('--labels')
    parser.add_argument('--save_embeddings_to')
    parser.add_argument('--save_concepts_to')
    parser.add_argument('--synsets')
    args = parser.parse_args()
    with open(args.synsets, encoding='utf-8') as input_stream:
        synsets = [json.loads(line) for line in input_stream]
    synsets = [synset for synset in synsets if len(synset) > 5]
    print("Loaded synsets")
    concepts = [s for synset in synsets for s in synset]
    entities = pd.read_csv(args.entities, sep='\t', names=['entity'])
    labels = pd.read_csv(args.labels, sep='\t', names=['label'])
    label_vocab = labels.label.drop_duplicates()
    label_vocab = [label  for label in label_vocab if label in concepts]
    concept2index = {label: label_id for label_id, label in enumerate(label_vocab)}

    word2index, weights_matrix = load_embeddings("/root/DATA/EMBEDDINGS/Health_2.5mreviews.s200.w10.n5.v15.cbow.bin")

    tokens_count = len(word2index)
    entities_count = len(concept2index)
    token_entity_distribution = get_token_entity_distribution(entities.entity.str.lower(), labels.label, word2index, concept2index) # np.random.rand(entities_count, tokens_count)
    token_entity_distribution = token_entity_distribution / token_entity_distribution.sum(axis=1, keepdims=True)
    #print(token_entity_distribution.sum(axis=1))
    token_distribution = get_token_distribution(entities.entity.str.lower(), word2index) # np.random.rand(tokens_count)
    token_distribution = token_distribution / token_distribution.sum()


    wordembedder = Embedder(word2index, weights_matrix)
    entity_embedder = Embedder(concept2index, input_size=len(concept2index), hidden_size=weights_matrix.shape[1])
    cuda = torch.device('cuda')
    wordembedder.to(cuda)
    entity_embedder.to(cuda)

    dataset = []
    for rep in range(5):
        for entity_id in range(len(concept2index)):
        #for rep in range(50):
            pos_word_id, neg_word_id = sample_pos_neg_words(entity_id, token_entity_distribution, token_distribution)
            dataset.append([entity_id, pos_word_id, neg_word_id])
    print(len(dataset))

    dataset = torch.LongTensor(dataset)
    dataloader = DataLoader(dataset, batch_size=32)
    embeddings_old = []

    e_optimizer = optim.SGD(entity_embedder.parameters(), lr=0.01, momentum=0.9)
    w_optimizer = optim.SGD(wordembedder.parameters(), lr=0.01, momentum=0.9)
    for batch in tqdm(dataloader):
        batch = batch.to(cuda)
        e_optimizer.zero_grad()
        w_optimizer.zero_grad()
        entity_embeddings = entity_embedder(batch[:, 0])
        pos_e = wordembedder(batch[:, 1])
        neg_e = wordembedder(batch[:, 2])
        d_pos = TripletDistanceMetric.COSINE(entity_embeddings, pos_e)
        d_neg = TripletDistanceMetric.COSINE(entity_embeddings, neg_e)

        loss = F.relu(d_pos - d_neg).mean()
        #print(loss)
        loss.backward()
        e_optimizer.step()
        w_optimizer.step()
        #wordembedder.zero_grad()
        #entity_embedder.zero_grad()

    wordembedder.eval()
    entity_embedder.eval()

    concepts = []
    embeddings = []
    with torch.no_grad():
        for concept, concept_id in concept2index.items():
            concept_id = torch.LongTensor([concept_id]).to(cuda)
            embedding = entity_embedder(concept_id).cpu().numpy()
            concepts.append(concept)
            embeddings.append(embedding)


    with open(args.save_concepts_to, 'w', encoding='utf-8') as output_stream:
        for concept in  concepts:
            output_stream.write(str(concept) + '\n')
    embeddings = np.vstack(embeddings)
    np.save(args.save_embeddings_to, embeddings)
