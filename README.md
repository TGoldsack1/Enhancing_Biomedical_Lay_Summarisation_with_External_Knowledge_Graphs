# Enhancing_Biomedical_Lay_Summarisation_with_External_Knowledge_Graphs

This repository will contain the code and data for the paper "[Enhancing Biomedical Lay Summarisation with External Knowledge Graphs](https://arxiv.org/abs/2310.15702)", accepted in EMNLP 2023.

## Running models with eLife data

### Downloading trained models
Our trained models can be downloaded automatically by running the `get_models.sh` script:

```
bash get_models.sh
```

### Downloading eLife graph data
Similarly, the eLife graph data can be downloaded by running the `get_elife_graph_data.sh` script:

```
bash get_elife_graph_data.sh
```

### Running generation with trained models

To generate summaries for the eLife data using our trained models, run the `generate.py` script with the path to the model you wish to use:

```
python generate.py {model_path}
```

### Running training with eLife data
To train a model on the eLife data, run the `train.py` script with the path a config file (see `train_configs.json` for an example):

```
python train.py {config_path}
```

## Running models with new data

In order to run any of the models on new data, new graph data files (in the same format as our eLife graph data) will need to be created.

Our graph data is stored in `.pkl` files (one for each data split), each of which is created from `.jsonl` file containing a list of dictionaries (one for each article) in the following format:

```
{
  "id": str,                                         # unique identifier
  "nodes": [node_id],                                # list of graph nodes, represented by their string ids
  "edges": [[src_node_id, rel_id, tgt_node_id]],     # list of relation tuples, represented by their node/relation string ids 
  "nfeatures": [node_embedding],                     # list of initial node features, n-dimentional arrays
}
```



As covered in the paper, we use data from [UMLS](https://www.nlm.nih.gov/research/umls/index.html) to construct our graphs, which can only be accessed upon obtaining a license. Specifically, use the [Metamap](https://metamap.nlm.nih.gov/) tool to map text to UMLS concepts, we use the [UMLS Metathesaurus](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/index.html) to retrieve semantic types of concepts, and, finally, we use the [UMLS API](https://documentation.uts.nlm.nih.gov/rest/home.html) to retrieve the definitions of the identified UMLS concepts and semantic types.

The code used to construct our graphs is located in the `graph_construction` directory.
