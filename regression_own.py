import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')
#xs= np.array([1,2,3,4,5],dtype=np.float64)
#ys= np.array([5,5,6,6,6],dtype=np.float64)

def create_dataset(n,variance,step=2,correlation=False):
    val=1
    ys=[]
    for i in range(n):
        y= val + random.randrange(-variance,variance)
        if correlation:
            val+= step
        else:
            val-= step
        ys.append(y)
    xs=[x for x in range(len(ys))]
    return np.array(xs,dtype=np.float64), np.array(ys,dtype=np.float64)


def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs*xs)))
    b= mean(ys) - (mean(xs)*m)
    return m,b

def best_fit_line(xs,ys,m):
    return b

def squared_error(y_orig,y_line):
    return sum((y_orig - y_line) **2)

def coefficient_of_determination(y_orig,y_line):
    y_mean_line= [mean(y_orig) for y in ys]
    squared_error_regr = squared_error(y_orig,y_line)
    squared_error_y_mean = squared_error(y_orig,y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

xs,ys= create_dataset(60,100,4,True)
m,b = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]


predict_x = 6
predict_y = (m*predict_x) + b

r_squared= coefficient_of_determination(ys,regression_line)
print(r_squared)
plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, regression_line)
plt.scatter(predict_x,predict_y,color='g',s=100)
plt.show()


# plt.show()
