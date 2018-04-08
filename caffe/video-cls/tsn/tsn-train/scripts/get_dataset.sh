#!/usr/bin/env bash

wget -O data/dataset.zip http://p22k53ufs.bkt.clouddn.com/tsn-dataset-splits.zip
unzip data/dataset.zip -d data/
rm data/dataset.zip