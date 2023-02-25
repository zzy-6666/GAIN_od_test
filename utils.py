# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''
 
# Necessary packages
import numpy as np
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#归一化
def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters,_代表临时变量，没什么用；dim代表dimension，即维度
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
#(1e-6是什么鬼???懂了，如果第i列恰好全部缺失，那么分母就变成0了，1e-6就是防止分母为0)
    # Return norm_parameters for renormalization返回每一列的最大最小值数组
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters

#重正化
def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data

#对分类型变量的值取整（分类型的变量值一般小于20个，且一般对应整数）
def rounding (imputed_data, data_x):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data

#均方根误差(原始数据与插补数据的误差)
def rmse_loss (ori_data, imputed_data, data_m):
  '''Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse: Root Mean Squared Error
  '''
  
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
    
  # Only for missing values，从掩码矩阵m中获取缺失数据的位置（用0代替，1-m则是用1代替）
  nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
  denominator = np.sum(1-data_m)
  
  rmse = np.sqrt(nominator/float(denominator))
  
  return rmse

#均方误差
# def mse_loss(ori_data, imputed_data, data_m):
#   ori_data, norm_parameters = normalization(ori_data)
#   imputed_data, _ = normalization(imputed_data, norm_parameters)
#   nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
#   denominator = np.sum(1 - data_m)
#
#   mse = nominator / float(denominator)
#   return mse
def mse_loss(ori_data, imputed_data, data_m):
  #data_m中缺失位为0
  nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
  denominator = np.sum(1 - data_m)

  mse = nominator / float(denominator)
  return mse

def mae_loss(ori_data, imputed_data, data_m):
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
  nominator = np.sum(np.absolute((1 - data_m) * ori_data - (1 - data_m) * imputed_data))
  denominator = np.sum(1 - data_m)

  mae = nominator/float(denominator)
  return mae

def mape_loss(ori_data, imputed_data, data_m):
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
  a = ((1 - data_m) * ori_data - (1 - data_m) * imputed_data)
  c=(1 - data_m) * ori_data
  c[c==0]=1
  b = a/c
  nominator = np.sum(np.absolute(b))
  denominator = np.sum(1 - data_m)

  mape = nominator / float(denominator)
  return mape

#随机生成矩阵
def xavier_init(size):
  '''Xavier initialization.
  Xavier初始化，为了解决随机初始化的问题提出来的一种初始化方法，思想就是尽可能的让输入和输出服从相同的分布
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.（初始化的随机向量）
  '''
#（？？？,size这里的输入是什么样的，大概就是我认为的那种（n,m）的形式，就是不清楚标准差xacier_stddev为什么这么算，为什么叫这个，暂时只需要知道是一种矩阵初始化的方式）
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)#根据输入样本的行数，确定标准差，但这个计算公式是哪里的
  return tf.random_normal(shape = size, stddev = xavier_stddev)
      
#离散式采样器，随机生成一个指定个数从0-1均匀分布中抽样出的数组成的矩阵，并转化为每一个变量与p的比较结果（0或1）组成的矩阵
def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix

#设置种子的binary_sampler
def binary_sampler_seed(p, rows, cols):
  '''Sample binary random variables.

  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns

  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  np.random.seed(1)
  unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
  binary_random_matrix = 1 * (unif_random_matrix < p)
  return binary_random_matrix

#均匀采样器，从指定上下界均匀分布中随机采样指定尺寸的矩阵并返回
def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])       

#每次调用，返回随机打乱的全部样本中的前batchsize个样本的索引编号,等价于随机取了从title个样本序号中随机抽取size个序号
def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx
  
# if __name__ == '__main__':
#   # a=np.array([[1,2,3],[4,5,6]])
#   # b=np.array([[1,2,3],[4,6,6]])
#   # c=np.array([[1,2,3],[4,9,6]])
#   # m=np.array([[1,1,1],[1,0,1]])
#   # print(mape_loss(a,b,m))
#   # print(mape_loss(a,c,m))
#   np.random.seed(1)
#   unif_random_matrix = np.random.uniform(0., 1., size=[3, 3])
#   binary_random_matrix = 1 * (unif_random_matrix < 0.8)
#   print(unif_random_matrix)
