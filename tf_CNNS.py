from gensim.models import word2vec
import logging
import random
import math
import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

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

print("读取test数据完毕！")
print("label_test是{}*{}矩阵".format(len(label_test),len(label_test[0])))
print("data_test是{}*{}矩阵".format(len(data_test),len(data_test[0])))
model = None


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 2500])
y_ = tf.placeholder(tf.float32, shape=[None, 9])

W_conv1 = weight_variable([7, 7, 1, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1,50,50,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([7, 7, 16, 64])       #Second Convolutional Layer
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * 64, 512])     #Densely Connected Layer
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)      #Dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512, 9])         #Readout Layer
b_fc2 = bias_variable([9])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train and Evaluate the Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
listnum = []
Iter = 0
temp_x = []
temp_y = []
pre_accuraccy1 = 0
pre_accuraccy2 = 0
test_accuracy = 0
for i in range(20000):
    _batch_size = 50
    if len(listnum) < _batch_size:
        listnum = list(range(len(data_train)))
        Iter += 1
    temp_x.clear()
    temp_y.clear()
    for _ in range(_batch_size):
        j = random.randint(0, len(listnum) - 1)
        temp_x.append(data_train[listnum[j]])
        temp_y.append(label_train[listnum[j]])
        del listnum[j]
    if (i+1)%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={ x:temp_x, y_: temp_y, keep_prob: 1.0})
        print("Iter %d:step %d, training accuracy %g"%(Iter,i+1, train_accuracy))
        test_accuracy = accuracy.eval(feed_dict={x: data_test, y_: label_test, keep_prob: 1.0})
        print("test accuracy %g" % (test_accuracy))
        if test_accuracy>0.835 or (test_accuracy+0.03<pre_accuraccy1 and pre_accuraccy1+0.03<pre_accuraccy2):
            break
        pre_accuraccy1 = test_accuracy
        pre_accuraccy2 = pre_accuraccy1
    train_step.run(feed_dict={x: temp_x, y_: temp_y, keep_prob: 0.5})
print("test accuracy %g"%(test_accuracy))

TPnum = [0,0,0,0,0,0,0,0,0]
FNnum = [0,0,0,0,0,0,0,0,0]
FPnum = [0,0,0,0,0,0,0,0,0]
TNnum = [0,0,0,0,0,0,0,0,0]
for i in range(len(data_test)):
    target = sess.run(tf.argmax(label_test[i],0))
    temp = [data_test[i]]
    pl = sess.run(tf.argmax(y_conv,1),feed_dict={x:temp, keep_prob:1.0})
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
    print("Class %d :Recall = %.3f , and Precision= %.3f  (TP= %d, FN= %d, FP= %d, TN= %d )"%(i,r,p,TPnum[k],FNnum[k],FPnum[k],TNnum[k]))
    pt += p
    rt += r
rt/=9
pt/=9
f1 = 2*pt*rt/(pt+rt)
print("Micro-R= %.3f ,Micro-P= %.3f , macro-F1= %.3f"%(rt,pt,f1))