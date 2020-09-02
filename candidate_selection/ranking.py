import numpy as np
from tqdm import tqdm
import faiss
import pandas as pd
import os


from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

from sentence_transformers import models
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.evaluation import TripletEvaluator
#from NlpOnTransformers.evaluation.SentenceEvaluator import SentenceEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.losses import TripletLoss
#from NlpOnTransformers.data_loading.ner_utils import save_embeddings, load_embeddings
#from NlpOnTransformers.losses.OnlineTripletLoss import OnlineTripletLoss


class RankinMapper:
    def __init__(self, model_args):
        self.model_args = model_args
        print("Loading Biobert")
        self.model = self.load_model(model_args['model_path'])
        print("Loading vocab")
        self.vocab, self.concept_ids_vocab = self.load_vocab(model_args['vocab_path'])
        self.index = None
        self.cpu_index = None
        self.search_count = self.model_args['search_count']
        self.threshold = self.model_args['threshold']
        print("Initializing Embeddings")
        self.init_embeddings()

    def init_embeddings(self):
        concept_embeddings_path = None
        if self.model_args['cache_dir'] is not None:
            concept_embeddings_path = os.path.join(self.model_args['cache_dir'], 'embeddings.h5')
        if concept_embeddings_path is not None and os.path.exists(concept_embeddings_path):
            print("Loading cached embeddings")
            concept_embeddings = load_embeddings(concept_embeddings_path).astype(np.float32)
        else:
            print("Calculating embeddings")
            concept_embeddings = self.embed(self.vocab)
        if concept_embeddings_path is not None and not os.path.exists(concept_embeddings_path):
            print("Caching embeddings")
            save_embeddings(concept_embeddings_path, concept_embeddings)
        print("Creating cpu index")
        self.cpu_index = faiss.IndexFlatL2(concept_embeddings.shape[1])
        if self.model_args['gpu']:
            print("Moving Index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.cpu_index)
        else:
            self.index = self.cpu_index
        print("Adding embeddings to index")
        self.index.add(concept_embeddings)

        # sanity check
        print("Sanity check")
        print(len(self.vocab), len(concept_embeddings))
        assert len(self.vocab) == len(concept_embeddings)
        print("Done loading ranking model")

    #@staticmethod
    def load_model(self, path):
        if self.model_args['w2v']:
            return KeyedVectors.load_word2vec_format(path, binary=True)
        checkpoint_files = os.listdir(path)
        if 'pytorch_model.bin' in checkpoint_files:
            word_embedding_model = models.BERT(path)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)
            return SentenceTransformer(modules=[word_embedding_model, pooling_model])
        return SentenceTransformer(path)

    def vectorize_text(self, text):
        embeddings = []
        for token in word_tokenize(text.lower()):
            if token in self.model:
                embeddings.append(self.model[token])
        if len(embeddings) > 0: return np.mean(embeddings, axis=0)
        return np.zeros(200)

    def embed(self, texts):
        if self.model_args['w2v']:
            return np.vstack([self.vectorize_text(t) for t in texts]).astype(np.float32)
        batch_size = 4096
        embeddinngs = []
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(len(texts), batch_start + batch_size)
            embeddinngs.append(self.model.encode(texts[batch_start:batch_end], show_progress_bar=True, batch_size=512))
        embeddinngs = np.concatenate(embeddinngs)
        return np.vstack(embeddinngs)

    @staticmethod
    def load_vocab(vocab_path):
        d = pd.read_csv(vocab_path, names=['concept_id', 'concept_name'], sep='\t').dropna()
        vocab = []
        concept_ids_vocab = []
        for line_id, line in d.iterrows():
            concept_id, concept_name = line['concept_id'], line['concept_name']
            vocab.append(concept_name.lower())
            concept_ids_vocab.append(concept_id)
        return np.array(vocab), np.array(concept_ids_vocab)

    def map_entities(self, data):
        if self.index is None:
            self.init_embeddings()
        entities = []
        #if len(entities) == 0: return data
        batch_size = 32
        embeddings_all = self.embed(data)
        embeddings_all = np.vstack(embeddings_all)
        predicted_labels = []
        for batch_start_idx in tqdm(range(0, len(data), batch_size), total=len(entities) // batch_size):
            #print(1)
            batch_end_idx = min(batch_start_idx + batch_size, len(data))
            embeddings = embeddings_all[batch_start_idx:batch_end_idx]
            batch_distances, batch_ids = self.index.search(embeddings, self.search_count)
            example = 0
            #print(batch_distances, batch_ids)
            for distances, ids in zip(batch_distances, batch_ids):
                top_concept_idxs = ids[np.argsort(distances)][:10]
                min_distances = np.sort(distances)[:10]
                #top_concept_idx = ids[np.argmin(distances)]
                #top_concept_ids = self.concept_ids_vocab[top_concept_idxs]
                #predicted_labels.append([top_concept_ids, float(distances.min())])
                extracted_entity = {
                    'entity_text': data[batch_start_idx + example],
                    'candidates': []
                }
                for concept_idx, score in zip(top_concept_idxs, min_distances):
                    concept_id = self.concept_ids_vocab[concept_idx]
                    concept_name = self.vocab[concept_idx]
                    extracted_entity['candidates'].append({
                        'concept_id': concept_id,
                        'mapped_to': concept_name,
                        'score': str(score),
                        'distance': None,
                    })
                #print(extracted_entity)
                entities.append(extracted_entity)
                example += 1
        return entities

    def eval(self, data):
        if self.index is None:
            self.init_embeddings()
        entities = []
        for document in data.documents:
            for entity in document.entities:
                # TODO add postprocessing
                entities.append(entity.text)
        batch_size = 256
        predicted_labels = []
        for batch_start_idx in tqdm(range(0, len(entities), batch_size), total=len(entities) // batch_size):
            batch_end_idx = min(batch_start_idx + batch_size, len(entities))
            embeddings = self.model.encode(entities[batch_start_idx:batch_end_idx], batch_size=128,
                                           show_progress_bar=False)
            embeddings = np.vstack(embeddings)
            batch_distances, batch_ids = self.index.search(embeddings, self.search_count)
            for distances, ids in zip(batch_distances, batch_ids):
                top_concept_idx = ids[np.argmin(distances)]
                top_concept_id = self.concept_ids_vocab[top_concept_idx]
                predicted_labels.append([top_concept_id, distances.min()])
        correct_predictions = 0
        entity_idx = 0
        for document in data.documents:
            for entity in document.entities:
                correct_predictions += (entity.label == predicted_labels[entity_idx][0])
                entity_idx += 1
        return {'acc@1': correct_predictions / entity_idx}

    def train(self, data_reader, experiment):
        train_data = SentencesDataset(data_reader.get_examples('train_neg15_pos15_hard.txt'), model=self.model)
        eval_data = SentencesDataset(data_reader.get_examples('train_neg15_pos15_hard.txt'), model=self.model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=experiment.batch_size)
        eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=experiment.batch_size)

        train_loss = TripletLoss(model=self.model)
        warmup_steps = int(len(train_data) / experiment.batch_size * experiment.epochs * 0.1)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=TripletEvaluator(eval_dataloader),
                       epochs=experiment.epochs,
                       evaluation_steps=500000,  # TODO
                       warmup_steps=warmup_steps,
                       output_path=experiment.output_dir
                       )

    def train_online_tl(self, data_reader, experiment):
        train_data = SentencesDataset(data_reader.get_examples('train_neg15_pos15_hard.txt'), model=self.model)
        eval_data = SentencesDataset(data_reader.get_examples('train_neg15_pos15_hard.txt'), model=self.model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=experiment.batch_size)
        eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=experiment.batch_size)

        train_loss = OnlineTripletLoss(model=self.model, vocab=self.vocab, concept_ids=self.concept_ids_vocab)
        warmup_steps = int(len(train_data) / experiment.batch_size * experiment.epochs * 0.1)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=SentenceEvaluator(),
                       epochs=experiment.epochs,
                       evaluation_steps=-1,  # TODO
                       warmup_steps=warmup_steps,
                       output_path=experiment.output_dir
                       )
