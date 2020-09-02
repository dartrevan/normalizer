from sentence_transformers import models
from sentence_transformers import SentenceTransformer, SentencesDataset
from gensim.models import KeyedVectors
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import os


def load_model(path):
    checkpoint_files = os.listdir(path)
    if 'pytorch_model.bin' in checkpoint_files:
        word_embedding_model = models.BERT(path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return SentenceTransformer(path)


def read_mrconso(fpath):
    """
    :param fpath: path to to MRCONSO.RRF file
    :return: DataFrame containing definitions of the concepts and synonyms
    """
    columns = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE',
               'STR', 'SRL', 'SUPPRESS', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8', usecols=['CUI', 'STR', 'LAT'])


def get_vector(cui_synonym_str):
    embeddings = []
    embedding_shape = w2v_model.vector_size
    for token in word_tokenize(cui_synonym_str):
        if token in w2v_model:
            embeddings.append(w2v_model[token])
    if len(embeddings) != 0:
        return np.mean(embeddings, axis=0)
    return np.zeros(embedding_shape)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--bert', action='store_true')
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--umls_ontology')
    parser.add_argument('--save_embeddings_to')
    parser.add_argument('--save_vocab_to')
    args = parser.parse_args()
    if not args.bert:
        w2v_model = KeyedVectors.load_word2vec_format(args.model_path, binary=args.binary)
    else:
        w2v_model = load_model(args.model_path)

    umls = read_mrconso(args.umls_ontology)
    umls = umls[umls.LAT == 'ENG'].dropna()

    if not args.bert:
        umls['embedding'] = umls.STR.apply(get_vector)
    else:
        umls['embedding'] = w2v_model.encode(umls.STR.tolist(), batch_size=128, show_progress_bar=True)
    umls_avg_embeddings = umls.groupby('CUI')['embedding'].apply(np.mean).reset_index()
    embeddings = np.vstack(umls_avg_embeddings.embedding.values)
    np.save(args.save_embeddings_to, embeddings)
    umls_avg_embeddings.CUI.to_csv(args.save_vocab_to, index=False, header=None, sep='\t', encoding='utf-8')

