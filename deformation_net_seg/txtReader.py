#coding:utf-8
import numpy as np


def read_txt(filename):
    idx = []
    pline = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() # 整行读取数据
            if not lines:
                break
                pass
            pline = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            idx.append(pline)  # 添加新读取的数据
            pass
        idx = np.array(idx).astype(np.int32) # 将数据从list类型转换为array类型。

        pass

    return idx

def write_text(filename, data):
    idx = []
    pline = []
    with open(filename, 'w') as file_to_write:
        for i in data:
            file_to_write.write(str(i)[1:-1]+'\r\n')  #\r\n为换行符


if __name__ == '__main__':
    '''M = np.zeros([19, 19])
    M[0, 1] = 1
    M[1, 2]= 1
    M[2, 3] = 1
    M[1,4] = 1
    M[1,7] = 1
    M[0,10] = 1
    M[0,13] = 1
    M[0,16] = 1
    M[4, 5] = 1
    M[5,6] = 1
    M[7, 8] = 1
    M[8,9] = 1
    M[10, 11] = 1
    M[11, 12] = 1
    M[13, 14] = 1
    M[14, 15] = 1
    M[16, 17] = 1
    M[17, 18] = 1

    idx = np.where(M > 0)
    edges = np.concatenate([np.reshape(idx[0],[-1,1]), np.reshape(idx[1],[-1,1])], axis=1)
    #idx = write_text()
    #print(idx)
    filename = '../Skeletons/horse_edges.txt'
    write_text(filename, edges)'''
    print(read_txt("../data_hdf5/seg.txt"))






