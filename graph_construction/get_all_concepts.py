import json

all_concepts = set()

ds = "eLife"

for split in ["train", "val", "test"]: 

  with open(f"./{ds}_split/{split}_disc_graphs_complete.jsonl", "r") as in_file:

    data = []
      
    for line in in_file.readlines():
      line_dict = json.loads(line)
      
      for n in line_dict['nodes']:
        
        if len(n) == 8 and n.startswith("C"):
          all_concepts.add(n)

with open("./umls_concepts_used.txt", "w") as out_file:
  for c in all_concepts:
    out_file.write(c)
    out_file.write("\n")