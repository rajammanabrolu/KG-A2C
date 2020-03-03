# KG-A2C
Goal driven language generation using knowledge graph A2C agents. This code accompanies the paper [Graph Constrained Reinforcement Learning for Natural Language Action Spaces](https://openreview.net/forum?id=B1x6w0EtwH).

Bibtex
```
@inproceedings{
ammanabrolu2020graph,
title={Graph Constrained Reinforcement Learning for Natural Language Action Spaces},
author={Prithviraj Ammanabrolu and Matthew Hausknecht},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=B1x6w0EtwH}
}
```
# Quickstart
Install Dependencies: Jericho, Redis, Pytorch >= 1.2
```bash
pip3 install --user jericho
pip3 install torch torchvision
sudo apt-get install redis-server
```

Download and extract [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/download.html) then start the OpenIE server:
```bash
cd stanford-corenlp-full-2018-10-05/ && java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

Train KG-A2C
```bash
cd kga2c && python train.py --rom_file_path path_to_your_rom --openie_path path_to_your_openie_install --tsv_file ../data/rom_name_here
```
