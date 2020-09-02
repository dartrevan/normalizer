from argparse import ArgumentParser
from es_mapper import ESMapper
#from ranking import RankinMapper
import json
import pandas as pd
import yaml


def load_config(path):
    with open(path) as config_input_stream:
        config = yaml.load(config_input_stream)
    return config

def acc_top_5(row):
    for candidate in row['candidates'][:5]:
        #print(candidate['mapped_to'], candidate['concept_id'], '|', row['label'], row['entity_text'],  str(candidate['concept_id']) == str(row['label']))
        #input()
        if str(candidate['concept_id']) == str(row['label']): return 1.0
    return 0.0


def acc_top_10(row):
    for candidate in row['candidates'][:10]:
        #print(candidate['mapped_to'], candidate['concept_id'], '|', row['label'], row['entity_text'],  str(candidate['concept_id']) == str(row['label']))
        #input()
        if str(candidate['concept_id']) == str(row['label']): return 1.0
    return 0.0


def acc_top_1(row):
    for candidate in row['candidates'][:1]:
        #print(candidate['mapped_to'], candidate['concept_id'], '|', row['label'], row['entity_text'],  str(candidate['concept_id']) == str(row['label']), candidate['score'])
        #input()
        if str(candidate['concept_id']) == str(row['label']): return 1.0
    return 0.0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--entities')
    parser.add_argument('--labels')
    parser.add_argument('--config_path')
    parser.add_argument('--mapper')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--lowercase', action='store_true')
    args = parser.parse_args()


    model_config = load_config(args.config_path)
    mapper = ESMapper(**model_config)
    #exit()
    entities = pd.read_csv(args.entities, sep='\t', names=['entity'], encoding='utf-8')
    labels = pd.read_csv(args.labels, sep='\t', names=['label'], encoding='utf-8')
    extracted_entities = mapper.map_entities(entities.entity.str.lower().tolist())

    extracted_entities = pd.DataFrame(extracted_entities)
    extracted_entities['label'] = labels.label
    print("Accuracy  top 1 is:", extracted_entities.apply(acc_top_1, axis=1).mean())
    print("Accuracy  top 5 is:", extracted_entities.apply(acc_top_5, axis=1).mean())
    print("Accuracy  top 10 is:", extracted_entities.apply(acc_top_10, axis=1).mean())

    extracted_entities = extracted_entities.to_dict('records')
    with open('test.txt', 'w', encoding='utf-8') as output_stream:
        for ee in extracted_entities:
            output_stream.write(json.dumps(ee) +  '\n')
