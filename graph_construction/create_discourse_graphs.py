import json
import logging
from os import path
from datetime import datetime
from enum import Enum
from umls import umls_search_concepts

logging.basicConfig(
  filename="./logs/create_discourse_graphs.log.{}".format(datetime.timestamp(datetime.now())), 
  level=logging.INFO,
  format = '%(asctime)s | %(levelname)s | %(message)s'
)

class Discourse_Relations(Enum):
  CONTAINS = "contains"
  HAS_TITLE = "has_title"
  HAS_KEYWORD = "has_keyword"
  WAS_PUBLISHED_IN = "was_published_in" # ref. year 
  
ISA_RELATION = 'T186'



def load_datafile(fp):
  with open(fp, "r") as in_file:

    data = json.loads(in_file.read())
    sections = [x["sections"] for x in data]
    section_names = [x['headings'] for x in data]
    abstracts = [x['abstract'] for x in data]
    titles = [x["title"] for x in data]
    keywords = [x["keywords"] for x in data]
    ids = [x["id"] for x in data]
    years = [x["year"] for x in data]
  
  return ids, sections, section_names, abstracts, titles, keywords, years


def get_discourse_graph(document_dict):
  nodes = set()
  edges = set()
  
  nodes.add(document_dict['id']) # document node

  # Title nodes / relations
  if document_dict['title'] != "":
    nodes.add(document_dict['title']) # title node
    edges.add((document_dict['id'], Discourse_Relations.HAS_TITLE.value, document_dict['title']))

  # Year nodes / relations
  if document_dict['year'] != "":
    nodes.add(document_dict['year'])
    edges.add((document_dict['id'], Discourse_Relations.WAS_PUBLISHED_IN.value, document_dict['year']))

  # Abstract nodes / relations
  if document_dict['abstract'] != "":
    abstract_node = document_dict['id'] + "_Abs"

    nodes.add(abstract_node)
    edges.add((document_dict['id'], Discourse_Relations.CONTAINS.value, abstract_node))

    abstract = " ".join(document_dict['abstract']).strip()

    # Abstract sentence nodes / relations
    kg_concepts = umls_search_concepts([abstract])[0][0]['concepts']

    for c in kg_concepts:
      nodes.add(c['cui'])
      edges.add((abstract_node, Discourse_Relations.CONTAINS.value, c['cui']))

      for stype in c['semtypes']:
        nodes.add(stype)
        edges.add((c['cui'], ISA_RELATION, stype))
        

  # Keyword nodes / relations
  for kw in document_dict['keywords']:
    nodes.add(kw)
    edges.add((document_dict['id'], Discourse_Relations.HAS_KEYWORD.value, kw))

  # Section nodes / relations
  for i, section in enumerate(document_dict['sections']):

    section_heading = document_dict['section_names'][i]
    
    # Section node
    sec_node = document_dict['id'] + "_Sec" + str(i)
    nodes.add(sec_node)
    
    edges.add((document_dict['id'], Discourse_Relations.CONTAINS.value, sec_node))

    # Section heading node
    nodes.add(section_heading)
    edges.add((sec_node, Discourse_Relations.HAS_TITLE.value, section_heading))

    kg_concepts = []
    for sent in section:
      kg_concepts.append(umls_search_concepts([sent])[0][0]['concepts'])

    for c in kg_concepts:
      nodes.add(c['cui'])
      edges.add((sec_node, Discourse_Relations.CONTAINS.value, c['cui']))

      for stype in c['semtypes']:
        nodes.add(stype)
        edges.add((c['cui'], ISA_RELATION, stype))

  return { 'edges': list(edges), "nodes": list(nodes) }


for fp in [
  "./resources/eLife_split/train.json",
  "./resources/eLife_split/val.json",
  "./resources/eLife_split/test.json",
]:

  ids, sections, section_names, abstracts, titles, keywords, years = load_datafile(fp)
  out_path = fp.replace(".json", "_disc_graphs.jsonl")
  is_existing = path.exists(out_path)
  o_type = "r+" if is_existing else "w"

  with open(out_path, o_type) as out_file:
    
    i = len(out_file.readlines()) if is_existing else 0
    
    for ind in range(i, len(ids)):
      logging.info(f'idx={ind}, id={ids[ind]}')

      data_dict = {
        "id": ids[ind],
        "sections": sections[ind],
        "section_names": section_names[ind],
        "abstract": abstracts[ind],
        "title": titles[ind],
        "keywords": keywords[ind],
        "year": years[ind],
      }

      out_dict = get_discourse_graph(data_dict)
        
      out_dict['id'] = ids[ind]
      out_file.write(json.dumps(out_dict))
      out_file.write("\n")
