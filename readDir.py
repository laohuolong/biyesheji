# -*- coding: UTF-8 -*-

'''
1、读取指定目录下的所有文件
2、读取指定文件，输出文件内容
3、创建一个文件并保存到指定目录
'''
import os

# 遍历指定目录，显示目录下的所有文件名
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
    print("共有{}文本在{}中.".format(len(fileList),filepath))
    return fileList


# 读取文件内容并打印
def readFile(filename):
    fopen = open(filename, 'r')  # r 代表read
    lines = fopen.readlines()
    temp =""
    for line in lines:
        temp += line.rstrip()
    fopen.close()
    return temp


# 输入多行文字，写入指定文件并保存到指定文件夹
def writeFile(openFileName,writeFileName):
    fileList = readDir(openFileName)
    fopen = open(writeFileName,'w')
    for file in fileList:
        fopen.write(readFile(file))
    fopen.close()


if __name__ == '__main__':
    openFileName = "E:\\sogou\\原数据\\train\\财经"
    writeFileName = "F:\\result.txt"
    writeFile(openFileName,writeFileName)