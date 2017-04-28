from gensim.models import word2vec
import logging
import math
import numpy as np

#  LSTM的文本分类
#  利用选取文档的词均分成50部分，每部分取词向量加和平均值作为一个词向量
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
model = word2vec.Word2Vec.load("dim50.model")
name = ["财经", "互联网", "健康", "教育", "军事", "旅游", "体育", "文化", "招聘"]
z50 = np.zeros(50)
maxlen = 50    #文档词向量个数以50取模，不足的以0补齐，若干个一起的均值作为一个向量，共50个，(设置过高，系统无法承受）
labels = [[1,0,0,0,0,0,0,0,0],
         [0,1,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,0],
         [0,0,0,1,0,0,0,0,0],
         [0,0,0,0,1,0,0,0,0],
         [0,0,0,0,0,1,0,0,0],
         [0,0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0,0,1]]
trainFileroot = "E:\\sogou\\分词后数据\\Reduced\\train\\"
testFileroot = "E:\\sogou\\分词后数据\\Reduced\\test\\"
data_train = []
label_train =[]
data_test = []
label_test =[]
output_data_test = open("E:\\sogou\\分词后数据\\Reduced\\result_lstm\\data_tf_test_vec.txt",'w')
output_label_test = open("E:\\sogou\\分词后数据\\Reduced\\result_lstm\\label_tf_test_vec.txt",'w')
output_data_train = open("E:\\sogou\\分词后数据\\Reduced\\result_lstm\\data_tf_train_vec.txt",'w')
output_label_train = open("E:\\sogou\\分词后数据\\Reduced\\result_lstm\\label_tf_train_vec.txt",'w')

def makeVec(fileroot,data,label):
    misnum = 0
    print("开始提取"+fileroot+"中的数据！")
    for i in range(len(name)):
        file = fileroot + name[i] + ".txt"
        file_object = open(file, encoding="UTF-8")
        line = file_object.readline()
        while line:
            vec = np.array([])
            flag = 0
            num = 0
            words = line.split()
            for word in words:
                try:
                    temp = model[word]
                except:
                    misnum += 1
                    if (misnum % 10000 == 0):
                        print("第" + str(misnum) + "个词：" + word + "not in vocabulary")
                else:
                    if flag > 0:
                        vec = np.vstack((vec, temp))
                    else:
                        vec = np.hstack((vec, temp))
                        flag = 1
                    num += 1
            n = math.floor(num / maxlen)
            k = num % maxlen
            _vec = np.array([])
            if n > 0:
                for ii in range(k):
                    temp = np.zeros(50)
                    for j in range(n + 1):
                        temp += vec[ii * (n + 1) + j]
                    temp /= (n + 1)
                    _vec = np.hstack((_vec, temp))
                pre = k * (n + 1)
                for ii in range(maxlen - k):
                    temp = np.zeros(50)
                    for j in range(n):
                        temp += vec[pre + ii * n + j]
                    temp /= n
                    _vec = np.hstack((_vec, temp))
            else:
                for ii in range(k):
                    _vec = np.hstack((_vec, vec[ii]))
                for _ in range(maxlen - k):
                    _vec = np.hstack((_vec, z50))
            label.append(labels[i])
            data.append(list(_vec))
            line = file_object.readline()
    print("读取train数据完毕！")
    print("label是{}*{}矩阵".format(len(label), len(label[0])))
    print("data是{}*{}矩阵".format(len(data), len(data[0])))

def writeVecToFile(fileroot,outputDataFile,outputLabelFile):
    misnum = 0
    print("开始提取数据！")
    for i in range(len(name)):
        file = fileroot + name[i] + ".txt"
        file_object = open(file, encoding="UTF-8")
        line = file_object.readline()
        print("开始读取"+file)
        while line:
            vec = np.array([])
            flag = 0
            num = 0
            words = line.split()
            for word in words:
                try:
                    temp = model[word]
                except:
                    misnum += 1
                    if (misnum % 10000 == 0):
                        print("第" + str(misnum) + "个词：" + word + "not in vocabulary")
                else:
                    if flag > 0:
                        vec = np.vstack((vec, temp))
                    else:
                        vec = np.hstack((vec, temp))
                        flag = 1
                    num += 1
            n = math.floor(num / maxlen)
            k = num % maxlen
            _vec = np.array([])
            if n > 0:
                for ii in range(k):
                    temp = np.zeros(50)
                    for j in range(n + 1):
                        temp += vec[ii * (n + 1) + j]
                    temp /= (n + 1)
                    _vec = np.hstack((_vec, temp))
                pre = k * (n + 1)
                for ii in range(maxlen - k):
                    temp = np.zeros(50)
                    for j in range(n):
                        temp += vec[pre + ii * n + j]
                    temp /= n
                    _vec = np.hstack((_vec, temp))
            else:
                for ii in range(k):
                    _vec = np.hstack((_vec, vec[ii]))
                for _ in range(maxlen - k):
                    _vec = np.hstack((_vec, z50))
            outputLabelFile.write(str(i) + "\n")
            _vec = list(_vec)
            for j in range(len(_vec)):
                outputDataFile.write("%.5f " % (_vec[j]))
            outputDataFile.write("\n")
            line = file_object.readline()


#makeVec(trainFileroot,data_train,label_train)
#makeVec((testFileroot,data_test,label_test))

writeVecToFile(trainFileroot,output_data_train,output_label_train)
writeVecToFile(testFileroot,output_data_test,output_label_test)