# KMLM
This is the source code of our method proposed in paper "Enhancing Multilingual Language Model with Massive Multilingual Knowledge Triples" accepted by EMNLP 2022.

# Code for Language Model Pretraining
Instructions can be found in ```k-mlm/README.md```.

# Wikidata Tools
Instructions can be found in ```wikidata_tool/README.md```.

# Data and Pretrained Models
Our data and checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1_IDaoNeCyYpEL1PCZdsQU5f7eRrOW-CN?usp=sharing).
- xlmr_base_kmlm.zip: The checkpoint of our base model (Mix).
- xlmr_large_kmlm.zip: The checkpoint of our large model (Mix).
- logic.zip: Dataset for the  cross-lingual logical reasoning (XLR) task.
- kg.zip & kg_clean.mono.zip: Multilingual synthetic sentences for knowledge oriented pretraining.
- kg.cycle.zip: Multilingual synthetic sentences for logical reasoning oriented pretraining. 

# Citation
Please cite our paper if you found the resources in this repository useful.
```
@inproceedings{liu-etal-2022-enhancing,
    title = "Enhancing Multilingual Language Model with Massive Multilingual Knowledge Triples",
    author = "Liu, Linlin  and
      Li, Xin and
      He, Ruidan and
      Bing, Lidong  and
      Joty, Shafiq  and
      Si, Luo",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = dec,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```
