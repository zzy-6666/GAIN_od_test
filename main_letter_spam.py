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

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import math

from data_loader import data_loader, test_loader
from gain import gain

### Command inputs:（输入内容）
'''
-   data_name: letter or spam
-   miss_rate: probability of missing components
-   batch_size: batch size
-   hint_rate: hint rate
-   alpha: hyperparameter
-   iterations: iterations
'''
### Example command（指令样例）
'''shell
$ 
python main_letter_spam.py --data_name spam --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 1000：0.0553 0.0542 0.0558，修正h后0.0546 0.0534
python main_letter_spam.py --data_name letter --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000：0.1304；1000:0.1364
python main_letter_spam.py --data_name huanghe --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 10 --iterations 10000
python main_letter_spam.py --data_name OD --batch_size 4 --hint_rate 0.6 --alpha 10 --iterations 10000
'''
### Outputs（输出内容）
'''
-   imputed_data_x: imputed data
-   rmse: Root Mean Squared Error
'''

def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  #生成原缺失数据所需参数如下
  data_name = args.data_name
  # miss_rate = args.miss_rate

  #输入到gain网络中的参数如下
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  '''
  *读取数据，读取训练数据在gain训练中进行，无需在此，此处只需读取测试集完整od，用于计算精度
  ————————————————————————————————————————————————————————————————
  '''
  # Load data and introduce missingness
  # ori_data_x, miss_data_x, data_m = data_loader()
  '''
  ————————————————————————————————————————————————————————————————
  '''

  imputed_data_x,D_loss,G_loss,mse = gain(gain_parameters)

  print('MSE Performance: ' + str(np.round(mse, 4)))
  # print('RMSE Performance: ' + str(np.round(rmse, 4)))
  # print('MAE Performance: ' + str(np.round(mae, 4)))
  # print('MAPE Performance: ' + str(np.round(mape, 4)))
  print("D_loss:",D_loss)
  print("G_loss:",G_loss)


  #标准差
  # rmse_std = math.sqrt(sum([(i-rmse)**2 for i in rmse_list])/len(rmse_list))
  # mse_std = math.sqrt(sum([(i - mse) ** 2 for i in mse_list]) / len(mse_list))
  # mae_std = math.sqrt(sum([(i - mae) ** 2 for i in mae_list]) / len(mae_list))
  # mape_std = math.sqrt(sum([(i - mape) ** 2 for i in mape_list]) / len(mape_list))

  #打印输出
  # print()
  # print('RMSE Performance: ' + str(np.round(rmse, 4)) + '± '+ str('%.4f' %rmse_std))
  # print('MSE Performance: ' + str(np.round(mse, 4)) + '± '+ str('%.4f' %mse_std))
  # print('MAE Performance: ' + str(np.round(mae, 4)) + '± '+ str('%.4f' %mae_std))
  # print('MAPE Performance: ' + str(np.round(mape, 4)) + '± '+ str('%.4f' %mape_std))
  # print()

  return imputed_data_x, mse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam','huanghe','OD'],
      default='spam',
      type=str)
  # parser.add_argument(
  #     '--miss_rate',
  #     help='missing data probability',
  #     default=0.2,
  #     type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  
  args = parser.parse_args() 


  # Calls main function  
  imputed_data, mse = main(args)