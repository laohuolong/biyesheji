# 读取文本所有内容，并以字符串列表返回
import math


def readFile(filename):
    fopen = open(filename,encoding="UTF-8")  # r 代表read
    lines = fopen.readlines()
    text =[]
    for line in lines:
        temp= line.rstrip().replace('&&&&','').split(" ")
        s = ''
        for word in temp:
            if isNum(word):
                s += (changeNumToN2(float(word))+" ")
            else:
                s += (word+" ")
        text.append(s)
    return text
    fopen.close()

def textToVec(documentDic,vec,filename):
    fopen = open(filename, encoding="UTF-8")  # r 代表read
    lines = fopen.readlines()
    textVec = []
    for line in lines:
        temp = {}
        wordList = line.rstrip().split(" ")
        wordDic = selecTextFreq(wordList,2)
        for word in wordDic:
            if word in documentDic:
                temp[vec[word]] = appro(wordDic[word] * math.log((1+ len(documentDic))/documentDic[word]),4)
        if len(temp) ==0:
            print(line)
        else:
            textVec.append(temp)
    return textVec

def appro(x,num):
    a = 1
    while num >0:
        a *=10
        num -=1
    return (int(x*a))/a

def selectWord(lines,threshold):
    documentDic = documentFreq(lines)
    if threshold <=1:
        return documentDic
    deleteList =[]
    for key in documentDic:
        if documentDic[key]<threshold:
            deleteList.append(key)
    for key in deleteList:
        del documentDic[key]
    return documentDic

def documentFreq(lines):
    documentDic = {}
    for line in lines:
        wordList = line.split(" ")
        wordDic = selecTextFreq(wordList,2)
        for word in wordDic:
            if word in documentDic:
                documentDic[word] +=1
            else:
                documentDic[word] = 1
    return documentDic

def selecTextFreq(wordList,threshold):
    wordDic = textFreq(wordList)
    if threshold <= 1:
        return wordDic
    deleteList = []
    for key in wordDic:
        if wordDic[key] < threshold:
            deleteList.append(key)
    if len(deleteList) == len(wordDic):
        print(wordDic)
        return wordDic
    for key in deleteList:
        del wordDic[key]
    return wordDic

def textFreq(wordList):
    wordDic ={}
    for word in wordList:
        if isNum(word):
            word = changeNumToN2(float(word))
        if word in wordDic:
            wordDic[word] +=1
        else:
            wordDic[word] = 1
    return wordDic

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
rootDir = "E:\\sogou\\分词后数据\\Reduced\\jieba\\train\\"
allText = open("E:\\sogou\\分词后数据\\Reduced\\jieba_traindata_tfidf.txt",'w',encoding="UTF-8")
lines =[]
for i in range(len(name)):
    filename = rootDir + name[i]+".txt"
    lines += readFile(filename)
print('读取lines完成')
documentDic = selectWord(lines,4)
print(len(documentDic))
print("select工作已完成")
vec ={}
flag =0
for key in documentDic:
    vec[key] = str(flag)
    flag += 1
for i in range(len(name)):
    filename = rootDir + name[i] + ".txt"
    textVec = textToVec(documentDic,vec,filename)
    for aText in textVec:
        allText.write(str(i) + ' ')
        for key in aText:
            allText.write(key+':'+str(aText[key])+' ')
        allText.write('\n')
    print(filename+"已读完！")
allText.flush()
allText.close()

