# Knowledge-based Multilingual Language Model Pre-training 
This is the code used to pretrain our knowledge-based multilingual language model.

## install packages
cd k-mlm; pip install . 

## example

1. modify the paths in ```bin/train_base_reason.sh```
2. specify the number of gpus in ```bin/acc_conf.yaml``` 
3. run command to pretrain

```
cd bin; bash train_base_reason.sh
```
