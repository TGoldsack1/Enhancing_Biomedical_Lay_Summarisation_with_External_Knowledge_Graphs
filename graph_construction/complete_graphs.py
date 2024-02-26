import json

ds = "eLife"

with open("./umls_semtypes.txt", "r") as in_file:
    lines = in_file.readlines()
    semtypes = {line.strip().split("|")[0]: line.strip().split("|")[1] for line in lines}

with open("./umls_rels.txt", "r") as in_file:
    lines = in_file.readlines()
    sem_rels = [line.strip().split("|") for line in lines]


def handle_dict(d):
    nodes = d['nodes']
    edges = d['edges']
    aid = d['id']

    new_nodes = []
    for n in nodes:

        # convert semantic type nodes to new format
        if n in semtypes:
            new_nodes.append(semtypes[n])
        else:
            new_nodes.append(n)

    # add semantic type relations
    for rel in sem_rels:
        n1, r, n2 = rel
        if n1 in new_nodes and n2 in new_nodes:
            edges.append([n1, r, n2])

    # convert semantic type nodes to UMLS format
    new_edges = set()
    for edge in edges:
        e1, r, e2 = edge

        if e2 in semtypes:
            e2 = semtypes[e2]
        
        if e1 in semtypes:
            e1 = semtypes[e1]

        new_edges.add((e1, r, e2))

    new_edges = list(new_edges)

    d['nodes'] = new_nodes
    d['edges'] = new_edges

    return d


for split in ["train", "val", "test"]: 

    out_file = open(f"./{ds}_split/{split}_disc_graphs_complete.jsonl", "w")

    with open(f"./{ds}_split/{split}_disc_graphs.jsonl", "r") as in_file:
            
        for i, line in enumerate(in_file.readlines()):
            line_dict = json.loads(line)
            d = handle_dict(line_dict)    
            out_file.write(json.dumps(d))
            out_file.write("\n")

    out_file.close()
