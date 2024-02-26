#!/bin/bash
# Author: Tomas Goldsack

# Start MetaMap server (once downloaded)
nohup ./public_mm/bin/skrmedpostctl start
nohup ./public_mm/bin/wsdserverctl start
sleep 60

# Run create initial graphs
nohup python create_discourse_graph.py

# Stop MetaMap server
/home/tomas/models/public_mm/bin/skrmedpostctl stop
/home/tomas/models/public_mm/bin/wsdserverctl stop