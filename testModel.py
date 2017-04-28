from gensim.models import word2vec
import logging
import numpy as np

#对词向量model进行测试

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
#model = word2vec.Word2Vec.load("dim100.model")
model = word2vec.Word2Vec.load("dim50.model")
while True:
    word = input("输入要查询的词")
    try:
        print(model[word])
    except:
        print(word+" not in vocabulary.")
    if word=="exit":
        break