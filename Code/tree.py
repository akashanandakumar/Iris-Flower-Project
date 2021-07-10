import numpy as np
import pandas as pd
from sklearn import datasets

data = datasets.load_iris()

names = data['target_names']

X = data['data']
X = np.array(X, dtype = int)
y = data['target']

data_ = pd.DataFrame(X)
data_.columns = data['feature_names']
data_['Output'] = Y

data['feature_names']

data_.isna().sum()

def entropy(col):
    
    counts = np.unique(col,return_counts=True)
    N = float(col.shape[0])
    
    ent = 0.0
    
    for ix in counts[1]:
        p  = ix/N
        ent += (-1.0*p*np.log2(p))
    
    return ent

def information_gain(x_data,key,val):
    left,right = divide_data(x_data,key,val)
    
    l = float(left.shape[0])/x_data.shape[0]
    r = float(right.shape[0])/x_data.shape[0]
    
    if left.shape[0] == 0 or right.shape[0] ==0:
        return -1000000 #Min Information Gain
    
    i_gain = entropy(x_data.Output) - (l*entropy(left.Output)+r*entropy(right.Output))
    return i_gain

def divide_data(x_data,key,val):
    #Work with Pandas Data Frames
    x_right = pd.DataFrame([],columns=x_data.columns)
    x_left = pd.DataFrame([],columns=x_data.columns)
    
    for ix in range(x_data.shape[0]):
        val_ = x_data[key].loc[ix]
        
        if val_ > val:
            x_right = x_right.append(x_data.loc[ix])
        else:
            x_left = x_left.append(x_data.loc[ix])
            
    return x_left,x_right

def find_count(X_train):                                   
        count = []
        count.append(X_train[X_train['Output'] == 0].shape[0])
        count.append(X_train[X_train['Output'] == 1].shape[0])
        count.append(X_train[X_train['Output'] == 2].shape[0])
        return count

class DecisionTree:
    
    def __init__(self,depth=0,max_depth=5):
        self.left = None
        self.right = None
        self.key = None
        self.val = None
        self.count = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None
    
    def train(self,X_train,names):
        
        features = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
        info_gains = []
        
        for ix in features:
            i_gain = information_gain(X_train,ix,X_train[ix].mean())
            info_gains.append(i_gain)
            
        self.key = features[np.argmax(info_gains)]
        self.val = X_train[self.key].mean()
        print("Level " , self.depth)
        self.count = find_count(X_train)
        cnt = 0                           
        for i in range(len(self.count)):
            if(self.count[i]):
                print("Count of " , names[i] , " = " , self.count[i])
                cnt += 1
        print("Current entropy = " , entropy(X_train.Output))
        if cnt != 1:
            print("Splitting on Tree Features ",self.key,"with information gain",np.argmax(info_gains))
        
        data_left,data_right = divide_data(X_train,self.key,self.val)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)
         
        if cnt == 1:
            if X_train.Output.mean() >= 1.5:
                self.target = names[2]
            elif X_train.Output.mean() <= 0.5:
                self.target = names[0]
            else:
                self.target = names[1]
            print("Reached leaf Node")
            print()
            print()
            return
        
        if(self.depth>=self.max_depth):
            if X_train.Output.mean() >= 1.5:
                self.target = names[2]
            elif X_train.Output.mean() <= 0.5:
                self.target = names[0]
            else:
                self.target = names[1]
            print("Max depth Reached")
            print()
            print()
            return
        
        print()
        print()
        
        self.left = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(data_left, names)
        
        self.right = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(data_right, names)
        
        if X_train.Output.mean() >= 1.5:
            self.target = names[2]
        elif X_train.Output.mean() <= 0.5:
            self.target = names[0]
        else:
            self.target = names[1]
        return    


dt = DecisionTree()    

dt.train(data_, names)     

