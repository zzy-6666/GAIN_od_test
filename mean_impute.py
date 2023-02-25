import numpy as np
import pandas as pd
# from data_loader import data_loader
#取每一列的均值（数值型变量）
def mean(miss_data_x):
    _, dim = miss_data_x.shape
    data = miss_data_x.copy()
    for i in range(dim):
        mean = np.average(miss_data_x[~np.isnan(miss_data_x[:, i]), i], axis=0)
        data[np.isnan(miss_data_x[:, i]), i]=mean
    return data

def mean_round(miss_data_x):
    _, dim = miss_data_x.shape
    data = miss_data_x.copy()
    for i in range(dim):
        mean = np.average(miss_data_x[~np.isnan(miss_data_x[:, i]), i], axis=0)
        data[np.isnan(miss_data_x[:, i]), i]=mean
        data[:,i]=np.round(data[:,i])
    return data

#取每一列的众数（分类型变量）
def most(miss_data_x):
    _, dim = miss_data_x.shape
    rounded_data = miss_data_x.copy().astype(int)
    for i in range(dim):
        counts=np.bincount(rounded_data[~np.isnan(miss_data_x[:, i]), i])
        max_num=np.argmax(counts)
        rounded_data[np.isnan(miss_data_x[:, i]), i]=max_num
    return rounded_data

def xx(miss_data_x):
    _, dim = miss_data_x.shape
    data=miss_data_x.copy()
    data_df=pd.DataFrame(data)
    for i in range(dim):
        data_df[i]=data_df[i].interpolate()
    data_ar=np.array(data_df)
    return data_ar
#



# if __name__ == '__main__':
    # ori_data_x, miss_data_x, data_m = data_loader('letter', 0.2)
    # mean(miss_data_x)
    # np.savetxt("imputed_data/mean_imputed.csv", miss_data_x, delimiter=",")
    # miss_data_x=np.array([[1,2,3],[1,2,6],[7,8,10],[np.NaN,np.NaN,np.NaN]])
    # print(miss_data_x)
    # a=mean_round(miss_data_x)
    # print(a)
    # miss_data_x=np.array([[1,2,3],[1,2,6],[np.nan,8,10],[np.NaN,np.NaN,np.NaN]])
    # print(miss_data_x[np.isnan(miss_data_x[:, 0]), 0])
    # print(np.isnan(miss_data_x[:, 0]))
    # a=miss_data_x[np.isnan(miss_data_x[:, 0]), 0]
    # print(a)

    # b=np.argwhere(np.isnan(miss_data_x[:, 0]))
    # print(b)
    # for i in b:
    #     if(i==0):
    #         miss_data_x[i]=
    #         a1=0
    #     else:
    #         a1=miss_data_x[i-1]
    #     if(i==len(miss_data_x[:,0])):
    #         miss_data_x[i] =miss_data_x[i]+1
    #     else:
    #         j=1
    #         while(np.isnan(miss_data_x[i+j]) and i+j<=len(miss_data_x[:,0])):
    #             j+=1
    #         a2=
    #         miss_data_x[i] = miss_data_x[i] + 1
    #
    # print(miss_data_x)

    # miss_data_x=np.array([[1,1,3],[2,2,6],[np.nan,3,10],[4,np.NaN,np.NaN]])
    # _, dim = miss_data_x.shape
    # data_df=pd.DataFrame(miss_data_x)
    # print(data_df)
    # for i in range(dim):
    #     data_df[i]=data_df[i].interpolate()
    # print(data_df)
    # data_ar=np.array(data_df)
    # print(data_ar)
