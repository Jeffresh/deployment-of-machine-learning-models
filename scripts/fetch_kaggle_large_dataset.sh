#!/usr/bin/env bash

TRAINING_DATA_URL="vbookshelf/v2-plant-seedlings-dataset"
NOW=$(date)

kaggle datasets download -d $TRAINING_DATA_URL -p section13/neural_network_model/neural_network_model/datasets/ && \
unzip section13/neural_network_model/neural_network_model/datasets/v2-plant-seedlings-dataset.zip -d section13/neural_network_model/neural_network_model/datasets/v2-plant-seedlings-dataset && \
echo $TRAINING_DATA_URL 'retrieved on:' $NOW > section13/neural_network_model/neural_network_model/datasets/training_data_reference.txt && \
mkdir -p "./section13/neural_network_model/neural_network_model/datasets/v2-plant-seedlings-dataset/Shepherds Purse"  && \
mv -v "./section13/neural_network_model/neural_network_model/datasets/v2-plant-seedlings-dataset/Shepherd's Purse/"* "./section13/neural_network_model/neural_network_model/datasets/v2-plant-seedlings-dataset/Shepherds Purse"
rm -rf "./section13/neural_network_model/neural_network_model/datasets/v2-plant-seedlings-dataset/Shepherd's Purse"