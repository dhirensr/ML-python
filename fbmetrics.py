import numpy as np
from sklearn import preprocessing,model_selection,neighbors
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,LogisticRegression
from adspy_shared_utilities import plot_fruit_knn,plot_feature_importances,plot_decision_tree
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score



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

def Weekday(x):
    if x == 1:
        return 'Su'
    elif x== 2:
        return 'Mo'
    elif x == 3:
        return 'Tu'
    elif x == 4:
        return 'We'
    elif x == 5:
        return 'Th'
    elif x ==6:
        return 'Fr'
    elif x == 7:
        return "Sa"

df['Weekday'] = df['Post Weekday'].apply(lambda x: Weekday(x))

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

#print(df.head())

dayDf = pd.get_dummies(df['Weekday'])
df = pd.concat([df,dayDf],axis=1)
hours = list(range(0,18))
#hours
for i in hours:
    hours[i] = str(hours[i])
    hours[i]='hr_'+ hours[i]

hourDf = pd.get_dummies(df['Post Hour'],prefix='hr_')
df = pd.concat([df,hourDf],axis=1)
monthDf = pd.get_dummies(df['Post Month'],prefix='Mo')
df = pd.concat([df,monthDf],axis=1)

df['Video'] = pd.get_dummies(df['Type'])['Video']
x = df[['Page total likes','Paid','Video','Status','Photo',
        'Cat_1','Cat_2','Mo','Tu','Sa',"We",'Th','Fr',
        'hr__17','hr__1','hr__2','hr__3','hr__4','hr__5', 'hr__6','hr__7','hr__8',
        'hr__9','hr__10','hr__11','hr__12','hr__13','hr__14','hr__15','hr__16','Mo_1',
        'Mo_2','Mo_12','Mo_4','Mo_5','Mo_6','Mo_7','Mo_8','Mo_9','Mo_11','Mo_10']]
y = df['like']

x_train,x_test,y_train, y_test = model_selection.train_test_split(x,
                                                                  y, test_size=0.1,
                                                                  random_state=42)

reg = LinearRegression(normalize=True)
lasso = Lasso(normalize=True)
reg.fit(x_train,y_train)
lasso.fit(x_train,y_train)

reg.fit(x_test,y_test)
#print(reg.score(x_test,y_test))

predicted_train = reg.predict(x_train)
predicted_test = reg.predict(x_test)
test_score = r2_score(y_test, predicted_test)
print(test_score)
#x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)
# clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8,
#                               random_state = 0).fit(x_train, y_train)



#clf= LinearRegression()

#cv_scores= model_selection.cross_val_score(clf,x,y)
# scaler= preprocessing.MinMaxScaler()
# scaler.fit(x_train)
# x_test= scaler.transform(x_test)
# x_train= scaler.transform(x_train)
#Clf=Lasso(alpha=8.0,max_iter=10000)
#clf= neighbors.KNeighborsRegressor(n_neighbors=5)
#print(cv_scores)

#clf.fit(x_train,y_train)

#accuracy= clf.score(x_test,y_test)
#print(accuracy*100)

# x_train.plot()
# y_train.plot()
# plt.show()
# clf= LinearRegression(normalize=True)
# clf.fit(x_train,y_train)

# sgd = SGDRegressor(penalty='elasticnet')
# sgd = sgd.fit(x,y)

#print(sgd.score(x_test,y_test))


#plt.figure(figsize=(10,4))
#plot_feature_importances(clf,['Page total likes','Post Month', 'Post Weekday',
          #                    'Post Hour', 'Paid','Photo','Video','Link','Status','Cat_1','Cat_2'])
#plt.show()
