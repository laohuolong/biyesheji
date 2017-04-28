from gensim.models import word2vec
import logging
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

#  LSTM的文本分类
#  利用选取文档的词均分成50部分，每部分取词向量加和平均值作为一个词向量

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
model = word2vec.Word2Vec.load("dim50.model")

name = ["财经", "互联网", "健康", "教育", "军事", "旅游", "体育", "文化", "招聘"]
z50 = np.zeros(50)
maxlen = 50    #文档词向量个数以50取模，不足的以0补齐，若干个一起的均值作为一个向量，共50个，(设置过高，系统无法承受）
label = [[1,0,0,0,0,0,0,0,0],
         [0,1,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,0],
         [0,0,0,1,0,0,0,0,0],
         [0,0,0,0,1,0,0,0,0],
         [0,0,0,0,0,1,0,0,0],
         [0,0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0,0,1]]

print("开始提取数据！")
fileroot = "F:\\data\\train\\"
data_train = []
label_train =[]
misnum =0
for i in range(len(name)):
    file = fileroot+name[i]+".txt"
    file_object = open(file,encoding="UTF-8")
    line = file_object.readline()
    train_num = 0
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
            else:
                if flag>0:
                    vec = np.vstack((vec,temp))
                else:
                    vec = np.hstack((vec, temp))
                    flag=1
                num+=1
        n = math.floor(num/maxlen)
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
        else:
            for ii in range(k):
                _vec = np.hstack((_vec, vec[ii]))
            for _ in range(maxlen-k):
                _vec = np.hstack((_vec, z50))
        label_train.append(label[i])
        data_train.append(list(_vec))
        line = file_object.readline()
        train_num +=1
    print("Class{}有{}训练样本数量.".format(i, train_num))
print("读取train数据完毕！")
print("label_train是{}*{}矩阵".format(len(label_train),len(label_train[0])))
print("data_train是{}*{}矩阵".format(len(data_train),len(data_train[0])))

fileroot = "F:\\data\\test\\"
data_test = []
label_test =[]
for i in range(len(name)):
    file = fileroot+name[i]+".txt"
    file_object = open(file,encoding="UTF-8")
    line = file_object.readline()
    test_num =0
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
            else:
                if flag>0:
                    vec = np.vstack((vec,temp))
                else:
                    vec = np.hstack((vec, temp))
                    flag=1
                num+=1
        n = math.floor(num/maxlen)
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
        else:
            for ii in range(k):
                _vec = np.hstack((_vec, vec[ii]))
            for _ in range(maxlen-k):
                _vec = np.hstack((_vec, z50))
        label_test.append(label[i])
        data_test.append(list(_vec))
        line = file_object.readline()
        test_num += 1
    print("Class{}有{}测试样本数量.".format(i,test_num))
print("读取test数据完毕！")
print("label_test是{}*{}矩阵".format(len(label_test),len(label_test[0])))
print("data_test是{}*{}矩阵".format(len(data_test),len(data_test[0])))
model = None

sess = tf.Session()
lr = 1e-2
batch_size = tf.placeholder(tf.int32)
input_size = 50
timestep_size = maxlen
hidden_size = 256
layer_num = 2
class_num = len(label_train[0])
keep_prob = tf.placeholder(tf.float32)

_X = tf.placeholder(tf.float32,[None,len(data_train[0])])
y = tf.placeholder(tf.float32,[None,class_num])
X = tf.reshape(_X,[-1,input_size,timestep_size])


lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=1.0,state_is_tuple=True)
lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prob)

mlstm_cell = rnn.MultiRNNCell([lstm_cell]*layer_num,state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size,dtype=tf.float32)

outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
h_state = state[-1][1]

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
listnum = []
Iter = 0
for i in range(2000):
    _batch_size = 90
    if len(listnum)<_batch_size:
        listnum = list(range(len(data_train)))
        Iter += 1
    temp_x =[]
    temp_y = []
    for _ in range(_batch_size):
        j = random.randint(0,len(listnum)-1)
        temp_x.append(data_train[listnum[j]])
        temp_y.append(label_train[listnum[j]])
        del listnum[j]
    if (i+1)%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            _X:temp_x, y: temp_y, keep_prob: 1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print("Iter %d: step %d, training accuracy %g" % ( Iter,(i+1), train_accuracy))
    sess.run(train_op, feed_dict={_X: temp_x, y: temp_y, keep_prob: 0.5, batch_size: _batch_size})
print("train accuracy %g"% sess.run(accuracy, feed_dict={_X:data_train, y: label_train, keep_prob: 1.0, batch_size:len(data_train)}))
print("test accuracy %g" % sess.run(accuracy, feed_dict={_X: data_test, y: label_test, keep_prob: 1.0,batch_size: len(data_test)}))

TPnum = [0,0,0,0,0,0,0,0,0]
FNnum = [0,0,0,0,0,0,0,0,0]
FPnum = [0,0,0,0,0,0,0,0,0]
TNnum = [0,0,0,0,0,0,0,0,0]
for i in range(len(data_test)):
    target = sess.run(tf.argmax(label_test[i],0))
    temp = [data_test[i]]
    pl = sess.run(tf.argmax(y_pre,1),feed_dict={_X:temp, keep_prob:1.0, batch_size:1})
    p = pl[0]
    for k in range(len(TPnum)):
        if target == k and p==k:
            TPnum[k] +=1
        elif target ==k and p!= k:
            FNnum[k] +=1
        elif target !=k and p==k:
            FPnum[k] +=1
        else:
            TNnum[k] +=1
pt =0
rt =0
for k in range(len(TPnum)):
    p = TPnum[k]/(TPnum[k] + FPnum[k])
    r = TPnum[k]/(TPnum[k] + FNnum[k])
    print("Class %d :Recall = %.3f , and Precision= %.3f  (TP= %d, FN= %d, FP= %d, TN= %d )" % (k, r, p, TPnum[k], FNnum[k], FPnum[k], TNnum[k]))
    pt += p
    rt += r
rt /= 9
pt /= 9
f1 = 2 * pt * rt / (pt + rt)
print("Micro-R= %.3f ,Micro-P= %.3f , macro-F1= %.3f" % (rt, pt, f1))
