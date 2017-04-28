from gensim.models import word2vec
import logging
import numpy as np

#将文档按词向量取加和平均得到的向量作为文档向量，用于svm模型的分类

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
#model = word2vec.Word2Vec.load("dim100.model")
model = word2vec.Word2Vec.load("dim50.model")
#fileroot = "F:\\data\\train\\"
fileroot = "F:\\data\\test\\"
name = ["财经", "互联网", "健康", "教育", "军事", "旅游", "体育", "文化", "招聘"]
#output=open("F:\\data\\pre_train_50.txt",'w')
output=open("F:\\data\\pre_test_50.txt",'w')
print(model)
misnum =0
dim =50
for i in range(len(name)):
    file = fileroot+name[i]+".txt"
    file_object = open(file,encoding="UTF-8")
    line = file_object.readline()
    while line:
        sum =np.zeros((1,dim))
        words = line.split()
        num = 1
        for word in words:
            try:
                temp=model[word]
            except:
                misnum+=1
            else:
                temp = temp.reshape((1, dim))
                sum += temp
                num+=1
        if num>1:
            num -=1
        sum/=num
        output.write(str(i)+" ")
        for j in range(dim):
            output.write("%d:%.4f "%(j,sum[0][j]))
        output.write("\n")
        line = file_object.readline()
output.flush()
output.close()
