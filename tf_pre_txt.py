from gensim.models import word2vec
import logging
import numpy as np


#lstm预处理文档
#将文档按100维词向量加和成平均词向量，当成该文档词向量表示方法。
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
model = word2vec.Word2Vec.load("dim100.model")
#fileroot = "E:\\sogou\\分词后数据\\Reduced\\train\\"
fileroot = "E:\\sogou\\分词后数据\\Reduced\\test\\"
name = ["财经", "互联网", "健康", "教育", "军事", "旅游", "体育", "文化", "招聘"]
#output_data = open("E:\\sogou\\分词后数据\\Reduced\\result_lstm\\data_tf_pre_train.txt",'w')
#output_label = open("E:\\sogou\\分词后数据\\Reduced\\result_lstm\\label_tf_pre_train.txt",'w')
output_data = open("E:\\sogou\\分词后数据\\Reduced\\result_lstm\\data_tf_pre_test.txt",'w')
output_label = open("E:\\sogou\\分词后数据\\Reduced\\result_lstm\\label_tf_pre_test.txt",'w')
misnum =0
for i in range(len(name)):
    file = fileroot+name[i]+".txt"
    file_object = open(file,encoding="UTF-8")
    line = file_object.readline()
    while line:
        sum =np.zeros((1,100))
        words = line.split()
        num=1
        for word in words:
            try:
                temp=model[word]
            except:
                misnum+=1
                if(misnum%1000 == 0):
                    print("第"+str(misnum)+"个词："+word+"not in vocabulary")
            else:
                temp = temp.reshape((1,100))
                sum += temp
                num+=1
        if num>1:
            num -=1
        sum/=num
        output_label.write(str(i)+"\n")
        for j in range(100):
            output_data.write("%f "%(sum[0][j]))
        output_data.write("\n")
        line = file_object.readline()
output_data.flush()
output_data.close()
output_label.flush()
output_label.close()
print(str(misnum))