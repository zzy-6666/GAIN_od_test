from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#文字
#
# # plt.title = ("走势图")
# # plt.xlabel = ("hint rate")
# # plt.ylabel = ("RMSE")

#提示率图1
# x=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
# y1=[0.18716277, 0.05368256, 0.05289486, 0.0731165, 0.068362266, 0.09104416, 0.0842162, 0.07051988, 0.07060002, 0.09497492]
# y2=[0.14634004, 0.10418503, 0.18941288, 0.2786439, 0.32266325, 0.38252804, 0.34738418, 0.30860484, 0.2578523, 0.39323884]
# y3=[0.20849116, 0.11604426, 0.25359732, 0.44186205, 0.45640182, 0.6082004, 0.5430175, 0.5431241, 0.52367043, 0.69167393]
# # y1=[0.8351275, 0.48856297, 0.45487309, 0.45373544, 0.4757617, 0.46344876, 0.4601298, 0.45207763, 0.42343912, 0.46259373]
# # y2=[0.67228943, 0.41410977, 0.32912904, 0.30647206, 0.30661106, 0.3015593, 0.30608892, 0.3107468, 0.3026717, 0.29723608]
# # y3=[0.9276473, 0.3688983, 0.2386852, 0.15043911, 0.14809784, 0.12370539, 0.116713956, 0.11083833, 0.11244787, 0.10047746]
# plt.plot(x,y1,label="0.2",color="red",marker="x")
# plt.plot(x,y2,label="0.5",color="green",marker="o")
# plt.plot(x,y3,label="0.8",color="blue",marker="+")
# # 绘制网格
# plt.grid(alpha=0.4)
# # 添加图例
# plt.legend(loc = 2)
# plt.show()

#缺失率图1
# x=[0.2,0.4,0.6,0.8]
# y1=[0.066,0.0853,0.1223,0.2434]
# y2=[0.0044,0.0073,0.0149,0.0593]
# y3=[0.0374,0.0544,0.079,0.1576]
# y4=[0.66,0.9311,1.3146,1.6494]
# plt.plot(x,y1,label="RMSE",color="red",marker='x')
# plt.plot(x,y2,label="MSE",color="green",marker='o')
# plt.plot(x,y3,label="MAE",color="blue",marker='+')
# # 绘制网格
# plt.grid(alpha=0.4)
# # 添加图例
# plt.legend(loc = 2)
# plt.show()

#缺失率图2
# x=[0.2,0.4,0.6,0.8]
# y1=[0.066,
# 0.0853,0.1223,0.2434]
# y2=[0.1325,0.1356,0.1348,0.1345]
# plt.plot(x,y1,label="GAIN",color="red",marker='x')
# plt.plot(x,y2,label="MEAN",color="blue",marker='+')
# # 绘制网格
# plt.grid(alpha=0.4)
# # 添加图例
# plt.legend(loc = 2)
# plt.show()

#提示率图2
# x=[0,0.2,0.4,0.6,0.8,1]
# y1=[0.08,0.0712,0.0709,0.0723,0.0853,0.1538]
# y2=[0.0064,0.0051,0.005,0.0052,0.0073,0.0236]
# y3=[0.048,0.0434,0.0408,0.0441,0.054,0.0993]
#
# plt.plot(x,y1,label="RMSE",color="red",marker='x')
# plt.plot(x,y2,label="MSE",color="green",marker='o')
# plt.plot(x,y3,label="MAE",color="blue",marker='+')
# # 绘制网格
# plt.grid(alpha=0.4)
# # 添加图例
# plt.legend(loc = 2)
# plt.show()

#对比柱状图
# x_data=['RMSE','MSE','MAE']
# y1_data=[0.066,0.0044,0.0374]
# y2_data=[0.1325,0.0175,0.0826]
# x_width=range(0,len(x_data))
# x2_width=[i+0.3 for i in x_width]
# plt.bar(x_width,y1_data,lw=0.5,fc="r",width=0.3,label="GAIN")
# plt.bar(x2_width,y2_data,lw=0.5,fc="b",width=0.3,label="MEAN")
# plt.legend(loc = 1)
# plt.xticks(range(0,3),x_data)
# plt.show()

# x_data=['RMSE','MSE','MAE']
# y1_data=[0.066,0.0044,0.0374]
# y2_data=[0.0272,0.0007,0.012]
# x_width=range(0,len(x_data))
# x2_width=[i+0.3 for i in x_width]
# plt.bar(x_width,y1_data,lw=0.5,fc="r",width=0.3,label="GAIN")
# plt.bar(x2_width,y2_data,lw=0.5,fc="b",width=0.3,label="MEAN")
# plt.legend(loc = 1)
# plt.xticks(range(0,3),x_data)
# plt.show()

# x_data=['RMSE','MSE','MAE']
# y1_data=[0.0526,0.0028,0.0208]
# y2_data=[0.0551,0.003,0.0224]
# x_width=range(0,len(x_data))
# x2_width=[i+0.3 for i in x_width]
# plt.bar(x_width,y1_data,lw=0.5,fc="r",width=0.3,label="GAIN")
# plt.bar(x2_width,y2_data,lw=0.5,fc="b",width=0.3,label="MEAN")
# plt.legend(loc = 1)
# plt.xticks(range(0,3),x_data)
# plt.show()

# x_data=['RMSE','MSE','MAE']
# y1_data=[0.1179,0.0147,0.0833]
# y2_data=[0.1354,0.0183,0.0838]
# x_width=range(0,len(x_data))
# x2_width=[i+0.3 for i in x_width]
# plt.bar(x_width,y1_data,lw=0.5,fc="r",width=0.3,label="GAIN")
# plt.bar(x2_width,y2_data,lw=0.5,fc="b",width=0.3,label="MEAN")
# plt.legend(loc = 1)
# plt.xticks(range(0,3),x_data)
# plt.show()

# x=['0.0001','0.001','0.01','0.1','1','10']
# x_width=range(0,len(x))
# y1=[0.0547,0.0549,0.0545,0.056,0.0585,0.0665]
# y2=[0.0543,0.054,0.0541,0.0541,0.0574,0.0631]
#
# plt.plot(x_width,y1,label="1000次",color="red",marker='x')
# plt.plot(x_width,y2,label="2000次",color="blue",marker='+')
# # 绘制网格
# plt.grid(alpha=0.4)
# # 添加图例
# plt.legend(loc = 2)
# plt.xticks(range(0,6),x)
# plt.show()