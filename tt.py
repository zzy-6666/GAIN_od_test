# import numpy as np
# import codecs
# import pandas as pd
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from tqdm import tqdm
# from utils import normalization, renormalization, rounding
# from utils import xavier_init
# from utils import binary_sampler, uniform_sampler, sample_batch_index

# print(np.zeros((3,3)))
# r=np.random.randint(0,5,(3,3))
# print(r)
#
# r1=np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(np.nanmax(r1[:,1]))
#
# print(1+1e-6)
#
# r2=np.array([[1,np.nan,3],[4,5,6],[7,8,9],[10,8,12]])
# print(np.isnan(r2[:,1]))
# print(r2[~np.isnan(r2[:,1]),1])
# print(r2[[True,True,True,True],1])
# print(np.unique(r2[~np.isnan(r2[:,1]),1]))
# print(len(np.unique(r2[~np.isnan(r2[:,1]),1])))
# r3=np.array([[1,np.nan,3],[4,5.5,6],[7,8,9],[10,8,12.1]])
# r3[:,1]=np.round(r3[:,1])
# print(r3)

# r4=np.random.rand(30,30)
# print(r4)
# print(len(np.unique(r4[:,1])))
# r4[:,1]=np.round(r4[:,1])
# print(r4)

# r5=np.array([[1,0],[0,1],[1,1]])
# print(r5)
# print(1-r5)
# print(np.sum(1-r5))

# print(4./2.)

# a = tf.Variable(tf.random_normal([3, 3], stddev=0.1))
# a = tf.Print(a, [a], "a: ",summarize=9)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# sess.run(a)

# size=(8,9)
# print(size[0])
# xavier_stddev=1. / tf.sqrt(size[0]/ 2.)
# print(tf.sqrt(8/2))
# with tf.Session() as sess:
#     norm_data=xavier_stddev.eval()
# print(norm_data)

# norm=tf.random_normal((4,3),mean=0.0, stddev=xavier_stddev, dtype=tf.float32)
# with tf.Session() as sess:
#     norm_data=norm.eval()
# print(norm_data[:])

# unif_random_matrix = np.random.uniform(0., 1., size = [3, 4])
# print(unif_random_matrix)
# binary_random_matrix = 1*(unif_random_matrix < 0.8)
# print(binary_random_matrix)

# total=40
# batch_size=20
# total_idx = np.random.permutation(total)
# batch_idx = total_idx[:batch_size]
# print(total_idx)
# print(batch_idx)



# no, dim = data_x.shape
# print(no,dim)
# miss_rate=0.1
# data_m = binary_sampler(1 - miss_rate, 3, 3)
# print(data_m)
#
# miss_data_x = data_x.copy()
# miss_data_x[data_m == 0] = np.nan
# print(data_m==0)
# print(miss_data_x)

# unif_random_matrix = np.random.uniform(0., 1., size = [2, 2])
# print(unif_random_matrix)
# binary_random_matrix = 1*(unif_random_matrix < (1-0.2))
# print(binary_random_matrix)
# print(binary_random_matrix==0)

# print(np.round(0.001,2))

# #定义缺失率，读取数据
# miss_rate=0.2
# file_name = 'data/'+'letter'+'.csv'
# data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
# print("data_x0:",data_x)
# print("data_x0.shape:",data_x.shape)
# #获取data_x缺失矩阵
# no, dim = data_x.shape
# data_m = binary_sampler(1 - miss_rate, no, dim)
# print("data_m0:",data_m)
# print("data_m0.shape:",data_m.shape)
# miss_data_x = data_x.copy()
# miss_data_x[data_m == 0] = np.nan
# data_x=miss_data_x
# print("data_x1:",data_x)
# print("data_x1.shape:",data_x.shape)
# #根据缺失矩阵获取掩码矩阵data_m（
# data_m = 1-np.isnan(data_x)
# print("data_m1:",data_m)
# print("data_m1.shape:",data_m.shape)
# #获取矩阵形状
# no, dim = data_x.shape
# #隐藏状态维度
# h_dim = int(dim)
# #原数据归一化，并获取归一化参数
# norm_data, norm_parameters = normalization(data_x)
# norm_data_x = np.nan_to_num(norm_data, 0)
# print("norm_data:",norm_data_x)
# print("norm_data.shape:",norm_data_x.shape)
# #定义变量
# D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
# D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
# print("D_W1:",D_W1)
# print("D_b1:",D_b1)

# h=uniform_sampler(0,0.01,3,3)
# print(h)
# p=0.8
# unif_random_matrix = np.random.uniform(0., 1., size=[2, 5])
# binary_random_matrix = 1 * (unif_random_matrix < p)
# print(binary_random_matrix)
# m=np.random.rand(2,5)
# print(m)
# h=m*binary_random_matrix
# h[binary_random_matrix==0]=0.5
# print(h)

# data_x = np.loadtxt('data/spam.csv', delimiter=",", skiprows=1)
# no, dim = data_x.shape
# data_m = binary_sampler(1-0.2,no,dim)
# miss_data_x = data_x.copy()
# miss_data_x[data_m == 0] = np.nan
# data_m = 1-np.isnan(miss_data_x)
# data_m=data_m.astype(np.float16)
# print(data_m.dtype)
# data_m[0,0]=0.5
# print(data_m[0,0])

# filecp = codecs.open('data/YRWL.csv', encoding = 'UTF-8')
# data_x = np.loadtxt(filecp, delimiter=",", skiprows=1,usecols=(0,1,2,3,4))
# print(data_x)

# data=pd.read_csv('data/YRWL.csv')
# data.loc[data['4']=='-','4'] = np.nan
# data['4'] = data['4'].astype(float)
# data['4'] = data['4'].interpolate()
# data.loc[data['5']=='-','5'] = np.nan
# data['5'] = data['5'].astype(float)
# data['5'] = data['5'].interpolate()
# data.to_csv('data/huanghe.csv', index=0, header=1)

# data_x = np.loadtxt('data/huanghe.csv', delimiter=",", skiprows=1)
# print(data_x)
# np.savetxt( "imputed_data/im.csv", data_x, delimiter="," )

# from matplotlib import pyplot as plt
# import random
#
# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
# plt.rcParams['axes.unicode_minus']=False
# plt.xlabel = ("Missing Rate (%)")
# plt.ylabel = ("RMSE")
# plt.title = ("对比")

# _xtick_labels1 = [i for i in range(0,81,10)]
# _xtick_labels2 = [i for i in range(0,5)]
# _xtick_labels3 = [i for i in range(0,81,10)]

# x = range(10,81,10)
# y1 = [0.01*random.randint(18,34) for i in range(8)]
# y2=[0.01*random.randint(18,34) for i in range(8)]
# y3=[0.01*random.randint(18,34) for i in range(8)]
#
#
# plt.plot(x,y1,label="gain",color="blue",marker='x')
# plt.plot(x,y2,label="MissForest",color="red",marker='o')
# plt.plot(x,y3,label="AutoEncoder",color="yellow",marker='+')
#
# plt.legend()
# plt.show()




'''
遗留问题:
1、超参数如何选择（应该在论文里，从0.1、0.5、1、2、10中通过交叉验证选择最佳，不过这种怎么进行交叉验证呢？？）
*2、为什么是数据集与掩码拼接来学习（通过一次训练，获取两种数据信息？拼接后相当于将掩码矩阵也为输入特征的后一部分，而这一部分与前一部分原始数据有着很大的相关性，通过神经网络提取这样的特征，从而生成能够骗过判别器的假数据，从而间接拟合真实分布，直接目的并不是生成拟合真实分布的数据）
    为什么提示矩阵h不直接用m，还要掩盖一些？（如果判别器一开始就知道m，）
3、会不会存在D强于G的情况，如何解决（按照训练中的输出来看，两者的loss值都是先减小，然后随机波动，不存在一方持续增大另一方持续减小的情况）
4、每次迭代先训练G会怎样（好像不会有太大问题，确实没什么，因为先训练G还是需要先把数据输入到G中进行生成，之后再输入到D中）
5、1e-6,1e-8有什么用（难道是防止一列数据中全是缺失值的情况？的确是这样）
6、rounding<20的值是为了将分类型变量转为整数，依据是判定这种变量一般小于20个，但我不理解为什么要round（哦哦因为原始数据是分类型变量的话，那么一定是整数，因为插入的值也必须是整数，如spam）
*7、网络为什么设计为3层，隐藏层的维度为什么由数据维度dim来定？
8、论文中的插图上，hint矩阵中的0.5是什么，依据代码的计算，hint中不是0就是1啊（代码有问题，现已经解决，但是为什么选择0.5？难道是概率问题？）
9、batchsize选多少为好（根据gpu的特性，以2的整数次幂最好，论文选择64）
*10、随机生成的z为什么上限是0.01，为什么不是1呢？
11、（变分）自编码器是什么原理

当前情况是：
1、论文中的数学看不懂（符号不确定、翻译不准确）
2、代码精读了一遍、能顺利跑通、能理解结构和过程，但是参数还没调整（还不清楚论文中的调参方法）、还没有在其他数据集尝试过
3、实验报告没写

准备：
1、先看下超参数是怎么调的（需要交叉验证）
2、试一下自己的数据
3、再次读论文
4、把随机梯度下降改成小批量梯度下降会怎样

成果：
1、根据论文发现了代码中存在的缺陷，并进行了修补，一定程度提升了精度

2、hint的作用：如果没有提示，那么判别器容易会将真实数据与，那么生成器生成的结果可能存在多种不同的分布，向判别器提示


'''



