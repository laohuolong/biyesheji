from gensim.models import word2vec
import logging

# 生成word2vec模型
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(u"F:\\data\\jieba_data.txt")  # 加载语料
model = word2vec.Word2Vec(sentences, size=50,min_count=3)  # 训练skip-gram模型，默认window=5
print(model)

