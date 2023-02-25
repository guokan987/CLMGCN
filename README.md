# CLMGCN
Due to our CLMGCN having two training steps, you should train 2 steps in your GPU as follows:

1. Contrastive learning pre-training phase (CLGCN):
run(or python) train.py --force True --model CLGCN --CL True  --data data/PEMS08  --adjdata data/PEMS08/adj_pems08.pkl --num_nodes 170 --save ./garage/pems08
 
Attention: when you training done, please empty GPU memory!

2. Prediction training phase (Base_GCN):
run(or python) train.py --force True --model Base_GCN --model1 CLGCN  --data data/PEMS08  --adjdata data/PEMS08/adj_pems08.pkl --num_nodes 170 --save ./garage/pems08

And also, we supply CLMGCN parameters of the trained model including the CLGCN (CL pretraining model) and Base_GCN (Prediction model) in the garage folders! and the data and garage are public by BaiduYun site:
