'''
需求：向gain随机提供一个明确形状的合成矩阵
返回：ori_data_x（完整OD）, miss_data_x(不完整OD), data_m（掩码）
'''

import numpy as np
import random
from utils import normalization

# 读取O、IOD、OD矩阵
def OD_loader(month=1, day=1, hour=0, part=1):
    # 根据文件名读取数据
    path = '../../taxi_Manhattan_2022_train/OD_2022_{}_{}_{}_{}.npz'.format(month, day, hour, part)
    data = np.load(path)
    iod = data['IOD']
    od = data['OD']
    o = data['O'].reshape(69, 1)
    # 获取标记矩阵m
    m = np.zeros([69, 69], dtype=float)
    m[od != iod] = 1
    # 归一化
    iod = normalization(iod)[0]
    od = normalization(od)[0]
    o = normalization(o)[0]
    # para3 = normalization(o)[1]
    return iod, od, o, m

# 随机批量抽取文件读取，作为训练集
def batch_sample(batch_nums=4):
    year = 2022
    data_list = []
    # 生成月、日、小时、粒度块对应的随机数
    for i in range(batch_nums):
        month = random.randint(1, 3)
        if month in [1, 3, 5, 7, 8, 10, 12]:
            d = 31
        elif month in [4, 6, 9, 11]:
            d = 30
        elif year % 4 == 0 and year % 100 != 0:
            d = 29
        else:
            d = 28
        #3月份后续作为测试集（20%），3月的第14天开始作为测试集
        if month==3:
            d=13
        day = random.randint(1, d)
        hour = random.randint(0, 23)
        # 1h = 4 * 15min
        part = random.randint(1, 4)
        # 读取数据并转换为data格式
        od_data = OD_loader(month, day, hour, part)
        data_list.append(od_data)
    return data_list

# 随机抽取1个文件
def one_sample():
    year = 2022
    month = random.randint(1, 3)
    if month in [1, 3, 5, 7, 8, 10, 12]:
        d = 31
    elif month in [4, 6, 9, 11]:
        d = 30
    elif year % 4 == 0 and year % 100 != 0:
        d = 29
    else:
        d = 28
    # 3月份后续作为测试集（20%）
    if month == 3:
        d = 13
    day = random.randint(1, d)
    hour = random.randint(0, 23)
    part = random.randint(1, 4)
    od_data = OD_loader(month, day, hour, part)
    return od_data

def one_sample_test():
    month = 3
    day = random.randint(14, 31)
    hour = random.randint(0, 23)
    part = random.randint(1, 4)
    od_data = OD_test_loader(month, day, hour, part)
    return od_data

# od_loader切换了文件路径
def OD_test_loader(month=3, day=14, hour=0, part=1):
    # 根据文件名读取数据
    path = '../../taxi_Manhattan_2022_test(0.2)/OD_2022_{}_{}_{}_{}.npz'.format(month, day, hour, part)
    data = np.load(path)
    iod = data['IOD']
    od = data['OD']
    o=data['O'].reshape(69,1)
    # 获取标记矩阵m
    m = np.zeros([69, 69], dtype=float)
    m[od != iod] = 1
    # m = torch.tensor(m)
    # 归一化
    iod = normalization(iod)[0]
    od = normalization(od)[0]
    o = normalization(o)[0]
    # 张量化
    # iod = torch.tensor(iod)
    # od = torch.tensor(od)
    # o = torch.tensor(o)
    # 类型转换
    # iod = iod.float()
    # od = od.float()
    # o=o.float()

    return iod, od, o, m

#返回测试集数据，需要合并后的iod，m，od，iod和m用于模型输入，od用于计算精度
def test_loader():
    month=3
    #读取第一组数据
    data=OD_test_loader()
    iod_data = data[0]
    od_data = data[1]
    m_data = data[3]
    m=1
    for day in range(14,32):
        for hour in range(24):
            for part in range(1,5):
                #读取并拼接所有测试集数据
                if m==1:
                    m=0
                    continue
                data = OD_test_loader(month, day, hour, part)
                iod_data = np.concatenate([iod_data, data[0]], axis=0)
                od_data = np.concatenate([od_data, data[1]], axis=0)
                m_data = np.concatenate([m_data, data[3]], axis=0)
    return iod_data, od_data, m_data

def data_loader():
    # Load data，每次读取一条od数据
    data=one_sample_test()
    data_x = data[1]
    miss_data_x = data[0]
    data_m = data[3]
    return data_x, miss_data_x, data_m

if __name__ == '__main__':
    a=test_loader()
    print(a[0].shape)
    print(a[1].shape)
    print(a[2].shape)