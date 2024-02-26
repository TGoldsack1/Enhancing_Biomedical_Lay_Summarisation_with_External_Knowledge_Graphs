## Graph construction

This directory contains the code used to construct our article knowledge graphs based on UMLS, which is partly based on that of the [bio relex](https://github.com/laituan245/bio_relex) - Joint Biomedical Entity and Relation Extraction with Knowledge-Enhanced Collective Inference.

Note that to run this code on any dataset other than eLife will likely require significant adjustments.

The creation of the graphs involves mutliple steps:

### 1. Creating the initial graph using MetaMap

To do this you will have to donwload MetaMap from the [UMLS website](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/Installation.html). The `run_create_graphs.sh` file can then be used to create the initial graphs.

Specifically, this `.sh` file launches the local MetaMap server (make sure that path to MetaMap is correct) and runs the `create_discourse_graphs.py` file to create the initial graphs, before stopping the MetaMap server.

The `create_discourse_graphs.py` builds the initial graphs for the articles in each split of eLife, the files for which are located in [this repo](https://github.com/TGoldsack1/Corpora_for_Lay_Summarisation).
The initial graphs consist of the elements extracted eLife (i.e., the central document node, section nodes, and metadata nodes) in addition to the concepts nodes identified by MetaMap.

### 2. Adding the semantic types and relations from UMLS

After constructing the initial graphs, we need to add the semantic types and relations that we collect from UMLS, alongside the definitions of semantic types and concepts that we use for node initialisation.

This can be done using the `complete_graphs.py` file. However, this file relies on `.txt` files containing lists of UMLS semantic types and relations (`umls_semtypes.txt` and `umls_rels.txt`, respectively) that cannot be shared via a public repository due to UMLS liscensing.

These files can be extracted directly from UMLS for which, you need to request [access permission](https://www.nlm.nih.gov/research/umls/index.html). If you want me to send you UMLS-related files, please email me at [tgoldsack1@sheffield.ac.uk](mailto:tgoldsack1@sheffield.ac.uk) (with some proof that you have access to UMLS).

### 3. Collecting definitions

To collect the definitions of the used semantic types and concepts, the `collect_definitions.py` file can be used. This file uses the UMLS API to collect the definitions of the semantic types and concepts that we use for node initialisation. Note that you will have to specifiy your UMLS API by editing the file.

The `collect_definitions.py` relies on both the `umls_semtypes.txt` file and another file containing the UMLS concepts you would like to collect definitions for (`umls_concepts_used.txt`). For the latter, the `get_all_concepts.py` file can be used to collect all the concepts used in the article graphs across all data splits.