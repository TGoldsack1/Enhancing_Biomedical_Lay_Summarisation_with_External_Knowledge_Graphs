#!/bin/bash

mkdir -p data

# Graph data
fileid="14cHpqW3b6upcCvwMUX-ZC1aFP4_0Zai5"
filepath="./data/graph_data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filepath}

unzip ./data/graph_data.zip -d ./data/