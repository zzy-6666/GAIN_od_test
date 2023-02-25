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

'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler
from data_loader import data_loader, test_loader
from utils import mse_loss


def gain (gain_parameters):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  #通过原数据获得掩码矩阵
  # data_m = 1-data_m
  
  # System parameters
  #系统读取输入的参数
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters (n*69,69)
  dim = 69

  # Hidden state dimensions(16)隐藏层维度（有什么用？？？即神经网络中隐藏层的维度；为什么直接设成dim）
  h_dim = int(dim)

  # Normalization归一化
  # norm_data, norm_parameters = normalization(data_x)
  # norm_data_x = np.nan_to_num(norm_data, 0)

  ## GAIN architecture   
  # Input placeholders（设定输入参数X,M,H的占位符，只会分配内存，但不会传入模型，等到session建立后，通过Session.run 的函数的 feed_dict 传入参数）
  # Data vector（None代表行数不定）
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables，判别器中的变量w，b，可以理解为三个全连接层，每个层都有w和b
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs，乘以2形成（32×16）是因为后续要将数据集和掩码矩阵拼接而成的矩阵（n×32）进行叉乘运算，产生n×16的中间矩阵
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables，生成器中的变量，和判别器同理
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  #生成器的前向传播，即由随机数和观测数拼接而成的矩阵，生成与真实值接近的矩阵
  def generator(x,m):
    # Concatenate Mask and Data，数据集和掩码矩阵拼接（横向
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output，sigmoid在这里起到归一化的作用
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  # 前向传播，由输入的矩阵和提示矩阵，输出对每个值真假的判定概率，组成的矩阵
  def discriminator(x, h):
    # Concatenate Data and Hint，数据集和提示矩阵拼接
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
 
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## GAIN loss
  #1e-8在这里是防止全部数据被判别器判断为0，导致log0的情况
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8))
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss

  ## GAIN solver，就是使得loss最小化的方法（优化器选用Adam）
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session() #TensorFlow中只有让Graph（计算图）上的节点在Session（会话）中执行，才会得到结果
  #tf.global_variables_initializer
  sess.run(tf.global_variables_initializer())#用于初始化所有变量，还可以用initializer初始化单个变量
   
  # Start Iterations，tqdm是一个进度条工具库
  D_loss_list=[]
  G_loss_list=[]
  for it in tqdm(range(iterations)):
    # Sample batch
    '''
    将随机抽取替换为随机读取,按照batchsize纵向组合多个数据
    ——————————————————————————————————————————————————————————————————
    '''
    #读取的第一个数据
    data = data_loader()
    data_m = 1 - data[2] #缺失位为0，非缺失位为1，从而帮助保留非缺失数据
    data_x = data[0] * data_m
    #后续batchsize-1个数据与前一个数据依次纵向拼接，最后大小为（batchsize*69,69）
    for i in range(batch_size - 1):
        data1 = data_loader()
        data_m1 = 1 - data1[2]
        data_x1 = data1[0] * data_m1
        data_x=np.concatenate([data_x, data_x1], axis=0)
        data_m=np.concatenate([data_m, data_m1], axis=0)
    X_mb = data_x
    M_mb = data_m

    # Sample random vectors
    '''
    z有无可能替换成不完整od？
    '''
    Z_mb = uniform_sampler(0, 0.01, batch_size * dim, dim) #随机生成0-0.01的size×dim的随机矩阵z，用于和观测数据x合成完整数据，注意每次迭代都会重新生成一个新的z

    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size * 69, dim)
    #与数据随机缺失处理是一个原理，hintrate代表提示信息的占有率，类似缺失信息的非缺失率（1-missing_rate），生成size×dim的0-1矩阵，
    #如果没有h，那么G会存在多种最佳拟合情况，加入h就能一定程度（取决于hint率）保证分布规律与真实数据分布的拟合，限制z的多变带来的不确定性
    H_mb = M_mb * H_mb_temp#即掩码数据中占比=hintrate的数据被保留，作为提示矩阵

    #源代码缺失的功能
    H_mb = H_mb.astype(np.float16)
    H_mb[H_mb_temp == 0] = 0.5 #提示矩阵中缺少的部分以0.5存在

    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 

    #开始训练，每次训练先训判别器D，在训练生成器G（若先训G，那么D就很难判别出G输出的真假了）
    #整个过程中，每次迭代X，M，H，Z都会随机改变，而G的w、b都会趋向于拟合真实数据的分布从而使得D越来越难以分辨真假
    _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})

    #用于输出loss曲线
    if((it+1)%1000==0):
        D_loss_list.append(D_loss_curr)
        G_loss_list.append(G_loss_curr)
        # print("D_loss_curr:",D_loss_curr,"  G_loss_curr:",G_loss_curr)
  ## Return imputed data
  '''
  插补，将该处数据替换为测试集数据：测试集全部读取，返回x和m，归一化，缺失位与随机数组合，输入至生成器，返回插补结果
  '''
  print("测试集插补中...")
  test_data_x = test_loader()
  no,dim=test_data_x[0].shape
  Z_mb = uniform_sampler(0, 0.01, no, dim) #（我更换了起始的随机矩阵，那么输入生成器模型的数据就改变了，那么模型中的参数w、b还会适用吗？）
  M_mb = 1-test_data_x[2]
  X_mb = test_data_x[0]
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  imputed_data_x = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  
  imputed_data_x = X_mb * M_mb + (1-M_mb) * imputed_data_x
  
  # Renormalization
  # imputed_data = renormalization(imputed_data, norm_parameters)
  
  # Rounding
  # imputed_data = rounding(imputed_data, data_x)

  #计算MSE
  ori_data_x = test_data_x[1]
  data_m = M_mb
  mse = mse_loss(ori_data_x, imputed_data_x, data_m)
  return imputed_data_x,G_loss_list,D_loss_list,mse

#数据预处理（x、m、z、h）（x和m随机抽取，z随机产生，h在m中随机抽取，最后由x和z组合，再与w组合交给G，将G产生的结果和h组合交给D）
#定义判别器和生成器的网络结构
#定义各自损失函数
#在迭代器中执行前向传播和随机梯度下降
#用生成器前向传播生成最终补齐的数据
