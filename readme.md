# polar-position-embedding demo

## Tow steps to run the Text Classification demo:
1. build the word dict file

select dataset in utils.py and run it:
```python
python utils.py
```

2. train the polar Transformer model:
```python
python train.py -b=32
```

## Tips
1. We build our model based on Pytorch 1.1.
2. The details of the model configs can be found by referring to the 'cof' in train.py.

## Configs in train.py
```json
cof = {"lr":8e-5, # learning rate
  "attention_probs_dropout_prob": 0.1, # dropout rate
  "rotate_lr": 0.2, # polar learning rate \gamma
  "hidden_act": "gelu", # active function
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768, # embedding size
  "num_attention_heads": 12, # head number of multi-head attention
  "num_hidden_layers": 12, # the number of layers of the model
  "type_vocab_size": 2, # the number of class types of classification task
  "vocab_size": 18784, # vocabulary size
  "adam_beta1": 0.9, # adam parameter 
  "adam_beta2": 0.999, # adam parameter 
  "weight_decay": 0.01, # L2 norm
  "n_warmup_steps": 10000, # Learning rate decline step
  "log_freq": 30 # n step for printing log Information
}
```

## test
For testing the trained model on test set, selece the datasets and run the script eval_acc.py:
```python
# selece dataset and  set 'test=True'
corpus = Corpus('subj', test=True)
```
```bash
# run the scipt
python eval_acc.py
```

## Our implementation of the "ENCODING WORD ORDER IN COMPLEX EMBEDDINGS, ICLR 2020" and other baselines
We have implemented the CVP model that is proposed in the paper "ENCODING WORD ORDER IN COMPLEX EMBEDDINGS" in ICLR2020. The source code is in the 'cvp' dictionary. You can easily run a cvp demo by replacing the "model_loc" dictionary  in the root dictionary by the "model_loc" dictionary in "cvp" dictionary. 

And you can also use the official implementation of the CVP in <a href ='https://github.com/iclr-complex-order/complex-order'>https://github.com/iclr-complex-order/complex-order</a>
