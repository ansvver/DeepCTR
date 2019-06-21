'''
# coding: utf-8
import matplotlib.pyplot as plt

#定义函数来显示柱状上的数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))


if __name__ == '__main__':
    l1=[68, 96, 85, 86, 76,87, 95]
    l2=[85, 68, 79, 89, 94, 82,90]

    name=['A','B','C','D','E','F','E']
    total_width, n = 0.8, 2
    width = total_width / n
    x=[0,1,2,3,4,5,6]

    a=plt.bar(x, l1, width=width, label="数学",fc = 'y')
    for i in range(len(x)):
      x[i] = x[i] + width
    b=plt.bar(x, l2, width=width, label="语文",tick_label = name,fc = 'r')

    autolabel(a)
    autolabel(b)

    plt.xlabel('学生')
    plt.ylabel('成绩')
    plt.title('学生成绩')
    plt.legend()
    plt.show()

'''

import numpy as np
import matplotlib.pyplot as plt

# x= [1,2,3,4,5,6,7,8]                                #[1]X坐标点
# y= [2,3,4,6,2,3,8,6]

plt.figure()
plt.subplot()
data= np.array([[3,3,3,15,3,3,3,10],         #[1]蓝色y坐标值
                       [2,6,2,3,2,9,14,2],          #[2]红色y坐标值
                       [4,3,9,4,12,4,21,4],         #[3]绿色y坐标值
                       [4,14,6,13,4,4,21,4]])       #[4]灰色y坐标值
index = np.arange(data.shape[1])
print(index)
color_index= ['r','g','b','y']

for i in range(4):
    # j = i+1
    #index+2:是起始坐标点    #width是bar的宽度
    plt.bar(index+2,data[i],width=0.7,color=color_index[i],bottom=np.sum(data[:i],axis=0))

plt.show()


import pandas as pd
df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])

print(df)