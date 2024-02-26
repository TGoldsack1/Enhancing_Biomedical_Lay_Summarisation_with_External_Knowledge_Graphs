# Taken from BioRelex repo

import os
import dgl
import pickle
import sqlite3
import torch

from constants import *
from os.path import join
from pymetamap import MetaMap
from sqlitedict import SqliteDict
from utils import create_dir_if_not_exist


# Main Functions
def umls_search_concepts(sents, prune=False, filtered_types = MM_TYPES):
    create_dir_if_not_exist(CACHE_DIR)
    search_results, cache_used, api_called = [], 0, 0
    sqlitedict = SqliteDict(UMLS_CONCEPTS_SQLITE, autocommit=True)
    for sent_idx, sent in enumerate(sents):

        # Use MetaMAP API
        api_called += 1
        METAMAP = MetaMap.get_instance(METAMAP_PATH)
        if not prune:
            raw_concepts, error = METAMAP.extract_concepts([sent], [0])
        else:
            raw_concepts, error = METAMAP.extract_concepts(
                [sent], 
                [0],
                4,
                None,
                'sldi',
                False,
                True, # word sense disambiguation
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                30,
                False,
                [],
                [],
                [],
                [],
                []
            )

        if error is None:
            sqlitedict[sent] = raw_concepts
        else:
            raise
        # Processing raw concepts
        processed_concepts = []
        for concept in raw_concepts:
            should_skip = False
            # Semantic Types
            if 'ConceptAA' in str(type(concept)):
                #print('Skipped ConceptAA')
                continue
            semtypes = set(concept.semtypes[1:-1].split(','))
            if len(semtypes.intersection(filtered_types)) == 0: continue # Skip
            semtypes = list(semtypes); semtypes.sort()
            # Offset Locations
            raw_pos_info = concept.pos_info
            raw_pos_info = raw_pos_info.replace(';',',')
            raw_pos_info = raw_pos_info.replace('[','')
            raw_pos_info = raw_pos_info.replace(']','')
            pos_infos = raw_pos_info.split(',')
            for pos_info in pos_infos:
                
                try:
                    start, length = [int(a) for a in pos_info.split('/')]
                except:
                    print('Skipped', pos_info)
                    start = 0
                    length = 0
                    continue
                
                start_char = start - 1
                end_char = start+length-1
                # Heuristics Rules
                concept_text = sent[start_char:end_char]
                if concept_text == 'A': continue
                if concept_text == 'to': continue
                # Update processed_concepts
                processed_concepts.append({
                    'cui': concept.cui, 'semtypes': semtypes,
                    'start_char': start_char, 'end_char': end_char,
                    "score": concept.score, "preferred_name": concept.preferred_name,
                    "trigger": concept.trigger
                })
        search_results.append({
            'sent_idx': sent_idx, 'concepts': processed_concepts
        })
    sqlitedict.close()
    return search_results, {'cache_used': cache_used, 'api_called': api_called}
