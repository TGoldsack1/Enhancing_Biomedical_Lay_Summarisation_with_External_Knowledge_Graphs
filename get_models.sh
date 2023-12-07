#!/bin/bash

mkdir -p models

# Models

## decoder_attn
fileid="1mAkmHfa97gtPdLHdMTtDQTO5uQrb8ZKI"
filepath="./models/decoder_attn.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filepath}

## doc_enhance
fileid="1u1dn33BBnQ7Pm0-15iOU_EBBV66Y4_io"
filepath="./models/doc_enhance.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filepath}

## text_aug
fileid="1EIRxSDxNd5Gej9LeW7sw33n7RPUxfMUB"
filepath="./models/text_aug.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filepath}

unzip ./models/decoder_attn.zip -d ./models/
unzip ./models/doc_enhance.zip -d ./models/
unzip ./models/text_aug.zip -d ./models/
