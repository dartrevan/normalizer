from elasticsearch import Elasticsearch
from tqdm import tqdm
from typing import Dict, Any

import spacy
import scispacy


class ESMapper:
    def __init__(self, index_name, vocab_path, spacy_model_name="en_core_sci_lg"):
        self.es = Elasticsearch()
        self.index_name = index_name
        self.vocab = self.load_vocab(vocab_path)
        self.nlp_spacy = spacy.load(spacy_model_name)

    def load_vocab(self, vocab_path):
        with open(vocab_path, encoding='utf-8') as input_stream:
            vocab = {}
            prev_doc_id = ''
            i = 0
            for doc_id, line in enumerate(input_stream):
                if '\t' in line:
                    doc_id, line = line.split('\t')
                if doc_id == prev_doc_id: i += 1
                else: i = 0
                prev_doc_id = doc_id
                vocab[str(doc_id) + '_{}'.format(i)] = line.strip().replace('"', '')
        return vocab

    def search_query(self, search_text: str, slop: int = 2) -> Dict[Any, Any]:
        #return {"query": {"term": {"text": search_text}}}
        return {"query": {"simple_query_string": {'query': search_text, 'fields': ["text"]}}}
        return {"query": {"match_phrase": {"text": {'query': search_text, 'slop': slop}}}}

    def extract_document_ids(self, search_text: str) -> Dict[Any, Any]:
        query = self.search_query(search_text)
        res = self.es.search(index=self.index_name, body=query, size=100,
                             sort='_score:desc')

        document_ids = {document['_id']: document['_score']
                        for document in res['hits']['hits']}
        #print(res['hits']['hits'])
        return document_ids

    def map_entities(self, data) -> Dict[Any, Any]:
        extracted_entities = []
        for entity in tqdm(data):
            extracted_entity = {
                'entity_text': entity,
                'candidates': []
            }
            concept_ids = self.extract_document_ids(entity)
            for concept_id, score in concept_ids.items():
                #concept_id = concept_id.split('_')[0]
                extracted_entity['candidates'].append({
                    'concept_id': concept_id.split('_')[0],
                    'mapped_to': self.vocab[concept_id],
                    'score': str(score),
                    'distance': None,
                })
            extracted_entities.append(extracted_entity)

        return extracted_entities
