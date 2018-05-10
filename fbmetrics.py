import numpy as np
from sklearn import preprocessing,model_selection,neighbors
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,LogisticRegression
from adspy_shared_utilities import plot_fruit_knn
import graphviz



style.use('ggplot')

df= pd.read_csv('dataset_Facebook.csv',sep=';')
#pre processing

df['Post Hour'].value_counts() # checks how many classes in paid
df['like'].fillna(0,inplace=True)
df['share'].fillna(0,inplace=True)
df['Paid'].fillna(0,inplace=True)

def Type(x,type_of):
    if x==type_of:
        return 1
    else:
        return 0

df['Photo'] = df['Type'].apply(lambda x: Type(x,'Photo'))
df['Video'] = df['Type'].apply(lambda x: Type(x,'Video'))
df['Link'] = df['Type'].apply(lambda x: Type(x,'Link'))
df['Status'] = df['Type'].apply(lambda x: Type(x,'Status'))
df['Cat_1'] = df['Category'].apply(lambda x: Type(x,1))
df['Cat_2'] = df['Category'].apply(lambda x: Type(x,2))

plotdf = df.drop(df.columns[7:15],axis =1)

fig, ax = plt.subplots()
paid = df[df['Paid']==1]
free = df[df['Paid']==0]  #seperated free users
#EDA

ax.scatter(paid['like'],paid['Lifetime Engaged Users'],color='b')
ax.scatter(free['like'],free['Lifetime Engaged Users'],color='y')
ax.set_title('Likes')
ax.set_xlim(0,1250)
ax.set_ylabel("# Users Reached")
ax.legend(labels=['Paid','Free'])
#plt.show()

x = df[['Page total likes','Post Month', 'Post Weekday',
        'Post Hour', 'Paid','Photo','Video','Link','Status','Cat_1','Cat_2']]
y = df['Lifetime Engaged Users']

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y, test_size=0.2)

# scaler= preprocessing.MinMaxScaler()
# scaler.fit(x_train)
# x_test= scaler.transform(x_test)
# x_train= scaler.transform(x_train)
#Clf=Lasso(alpha=8.0,max_iter=10000)
#clf= neighbors.KNeighborsRegressor(n_neighbors=5)

clf=LogisticRegression()
clf.fit(x_train,y_train)

accuracy= clf.score(x_test,y_test)
print(accuracy*100)

# x_train.plot()
# y_train.plot()
# plt.show()
# clf= LinearRegression(normalize=True)
# clf.fit(x_train,y_train)

# sgd = SGDRegressor(penalty='elasticnet')
# sgd = sgd.fit(x,y)

#print(sgd.score(x_test,y_test))
