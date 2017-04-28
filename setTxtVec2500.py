from gensim.models import word2vec
import logging
import numpy as np
import math

#将文档按词向量取加和平均得到的向量作为文档向量，用于svm模型的分类

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
#model = word2vec.Word2Vec.load("dim100.model")
model = word2vec.Word2Vec.load("dim50.model")
fileroot = "F:\\data\\train\\"
#fileroot = "F:\\data\\test\\"
name = ["财经", "互联网", "健康", "教育", "军事", "旅游", "体育", "文化", "招聘"]
output=open("F:\\data\\pre_train_2500.txt",'w')
#output=open("F:\\data\\pre_test_2500.txt",'w')
print(model)
misnum =0
dim =50
data_train = []
label_train =[]
maxlen = 50
z50 = np.zeros(50)
for i in range(len(name)):
    file = fileroot+name[i]+".txt"
    file_object = open(file,encoding="UTF-8")
    line = file_object.readline()
    while line:
        vec = np.array([])
        flag = 0
        num = 0
        words = line.split()
        for word in words:
            try:
                temp=model[word]
            except:
                misnum += 1
                if (misnum % 10000 == 0):
                    print("第" + str(misnum) + "个词：" + word + "not in vocabulary")
            else:
                if flag>0:
                    vec = np.vstack((vec,temp))
                else:
                    vec = np.hstack((vec, temp))
                    flag=1
                num+=1
        n = math.floor(num / maxlen)
        k = num % maxlen
        _vec = np.array([])
        if n>0:
            for ii in range(k):
                temp = np.zeros(50)
                for j in range(n+1):
                    temp += vec[ii*(n+1)+j]
                temp /= (n+1)
                _vec = np.hstack((_vec,temp))
            pre = k*(n+1)
            for ii in range(maxlen-k):
                temp = np.zeros(50)
                for j in range(n):
                    temp += vec[pre+ii*n+j]
                temp /= n
                _vec = np.hstack((_vec,temp))
            output.write(str(i) + " ")
            for j in range(len(_vec)):
                output.write("%d:%.4f " % (j, _vec[j]))
            output.write("\n")
        else:
            for ii in range(k):
                _vec = np.hstack((_vec, vec[ii]))
            output.write(str(i) + " ")
            for j in range(len(_vec)):
                output.write("%d:%.4f " % (j, _vec[j]))
            output.write("\n")
        line = file_object.readline()
output.flush()
output.close()
print(str(misnum))