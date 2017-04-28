import os
import random
import jieba
import jieba.analyse


# 如果有一些词语需要合并可以添加个人词典
# jieba.load_userdict('userdict.txt')

# 创建停用词列表,并以列表形式返回
def creadstoplist(stopwordspath):
    stwlist = [line.strip() for line in open(stopwordspath, 'r', encoding='utf-8').readlines()]
    return stwlist

#读取文件夹下的文本列表，并以列表形式返回
def readDir(filepath):
    pathDir = os.listdir(filepath)
    fileList =[]
    while len(pathDir)>0:
        dirName = pathDir.pop()
        temp = filepath+"\\"+dirName
        if os.path.isfile(temp):
            fileList.append(temp)
        elif os.path.isdir(temp):
            pathDir.append(temp)
    print("共有{}文本在{}中.".format(len(fileList),filepath))  # .decode('gbk')是解决中文显示乱码问题
    return fileList

# 读取文本所有内容，并以字符串返回
def readFile(filename):
    fopen = open(filename,encoding="UTF-8")  # r 代表read
    lines = fopen.readlines()
    temp =""
    for line in lines:
        temp += (line+" ")
    return temp
    fopen.close()

# 对句子进行分词，在停用词表中，且长度≤1的词会被舍弃掉,并返回分词后的字符串
def seg_sentence(sentence,stopwords):
    wordList = jieba.cut(sentence)
    outstr = ''
    for word in wordList:
        if word not in stopwords:
            if len(word) > 1:  # 去掉长度小于1的词
                if isNum(word):
                    word = changeNumToN2(float(word))
                if word != '\t':
                    word += " "
                    outstr += word
    return outstr

def isNum(word):
    try:
        float(word)
    except ValueError:
        return False
    else:
        return True

def changeNumToN2(num):
    if num<0:
        num = - num
    n = 0
    while num>=1:
        n+=1
        num /=10
    return 'N'+ str(n)

name = ["财经", "互联网", "健康", "教育", "军事", "旅游", "体育", "文化", "招聘"]
rootDir ="E:\\sogou\\原数据\\Reduced\\"
writeRoot = "F:\\data\\"
allText = open("F:\\data\\jieba_data.txt",'w',encoding="UTF-8")
stopWordsFile = "F:\\temp\\毕业设计总结\\去停用词.txt"
stopWords = creadstoplist(stopWordsFile)
r = 0.9 #会以r 和 1-r 的比例分为训练集 和测试集
for i in range(len(name)):
    fileDir = rootDir + name[i]
    writeTrain = open(writeRoot + "train\\"+name[i]+".txt",'w',encoding='UTF-8')
    writeTest = open(writeRoot +"test\\"+name[i]+".txt",'w',encoding='UTF-8')
    train = []
    test = []
    test_num = 190
    train_num = 1800
    fileList = readDir(fileDir)
    for file in fileList:
        rand = random.random()
        if train_num>0 and test_num>0:
            if rand<r:
                train.append(readFile(file))
                train_num -= 1
            else:
                test.append(readFile(file))
                test_num -= 1
        elif train_num==0:
            test.append(readFile(file))
            test_num -= 1
        else:
            train.append(readFile(file))
            train_num -= 1
    for text in train:
        text = text.replace("nbsp","").replace("\n","")
        temp = seg_sentence(text,stopWords)
        writeTrain.write(temp + '\n')
        allText.write(temp + '\n')
    for text in test:
        text = text.replace("nbsp", "").replace("\n", "")
        temp = seg_sentence(text, stopWords)
        writeTest.write(temp + '\n')
        allText.write(temp + '\n')
    writeTrain.flush()
    writeTest.flush()
    writeTrain.close()
    writeTest.close()
allText.flush()
allText.close()

