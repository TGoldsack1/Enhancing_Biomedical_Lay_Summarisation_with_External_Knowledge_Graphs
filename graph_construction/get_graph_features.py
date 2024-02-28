from transformers import AutoTokenizer, AutoModel
import pickle
import torch
import json
import torch.nn as nn
import re
from enum import Enum


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
if scibert_tokenizer.pad_token is None:
   scibert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

with open("./semtype_definitions.txt", "r") as in_file:
    semtype_defs = in_file.readlines()
    semtype_defs = {line.strip().split("|")[0]: line.strip().split("|")[-1] for line in semtype_defs}

with open("./umls_concept_definitions.txt", "r") as in_file:
    concept_defs = in_file.readlines()
    concept_defs = {line.strip().split("|")[0]: line.strip().split("|")[-1] for line in concept_defs}

def is_concept_node(node_id):
    return re.match(r'^[C][0-9]{7}', node_id)

def is_semtype_node(node_id):
    return re.match(r'^[T][0-9]{3}', node_id)

def get_node_type(node):
    if is_concept_node(node):
        return NodeType.Concept

    if is_semtype_node(node):
        return NodeType.Semtype

    if node.startswith("elife-"):
        if "_Abs" or "_Sec" in node:
            return NodeType.Section
        else:
            return NodeType.Document

    return NodeType.Metadata    


NodeType = Enum('NodeType', ['Document', "Section" 'Metadata', 'Concept', "Semtype"])


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(sents, is_project=False):
    sents = [s for s in sents if s]
    encoded_input = scibert_tokenizer(sents, padding='max_length', truncation=True, return_tensors='pt', max_length=250).to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = scibert(**encoded_input)

    # Perform pooling. In this case, mean pooling
    pool = mean_pooling(model_output, encoded_input['attention_mask'])

    if is_project:
        m = nn.Linear(768, 50).to(device)
        return(m(pool))
    else:
        return pool

def get_initial_embeddings(aid, nodes, edges):
    ret_nodes, ret_embs = [], []
    has_title_edges = [e for e in edges if e[1] == "has_title"]
    titles = [r[2] for r in has_title_edges]

    
    for n in nodes:
        if is_concept_node(n):           ## Concept nodes ##
            text = concept_defs[n]
            emb = get_sentence_embeddings([text], True)
            ret_embs.append(emb[0])
            ret_nodes.append(n)
        else:                            ## Non-concept nodes ##
            if is_semtype_node(n):
                text = semtype_defs[n]
                emb = get_sentence_embeddings([text], True)
                ret_embs.append(emb[0])
                ret_nodes.append(n)
            else:                        # titles, keywords, and section text
                if aid in n:
                    if "_Abs" in n:
                        title = "Abstract"
                    else:  
                        title_e = [e for e in has_title_edges if e[0] == n][0]
                        title = title_e[2]
                    if title.strip():
                        emb = get_sentence_embeddings([title], True)[0]
                        ret_embs.append(emb)
                        ret_nodes.append(n)
                else:                    # titles and keywords (titles ignored as used for node/section embeddings)
                    if n not in titles:
                        emb = get_sentence_embeddings([n], True)[0]
                        ret_embs.append(emb)
                        ret_nodes.append(n)                        
        
    return ret_nodes, ret_embs


ds = "eLife"
for split in ['train', 'val', 'test']:
    out_graphs = []

    with open(f"./{ds}_split/{split}_disc_graphs_complete.jsonl", "r") as in_file:
        graphs = [json.loads(line) for line in in_file.readlines()]

    for i, graph in enumerate(graphs):
        nodes = graph['nodes']
        edges = graph['edges']
        article_id = graph['id']

        final_nodes, embeddings = get_initial_embeddings(article_id, nodes, edges)
        final_edges = [e for e in edges if (e[0] in final_nodes and e[2] in final_nodes)]

        embeddings = torch.stack(embeddings).to("cpu").tolist()
        
        graph['nodes'] = final_nodes
        graph['edges'] = final_edges
        graph['nfeatures'] = embeddings
        out_graphs.append(graph)

    pickle.dump(out_graphs, open(f"./{ds}_split/{split}_graphs_with_features.pkl", 'wb'))
