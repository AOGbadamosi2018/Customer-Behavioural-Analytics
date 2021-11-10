import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Data.csv.csv")

#important to note that in importing files file exenstions should be included
X= dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

dataset.describe()

#Transformers for missing value imputation
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan,strategy='mean')
#since we don't want to transform the index,first wcolumn

imputer=imputer.fit(X[:,1:3]) 

#replace missing values woth mean

X[:,1:3] = imputer.transform(X[:,1:3])


#Encoding Categorical Data]
#
#
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[0])],remainder='passthrough')
##next fitting our data to the object

X=np.array(ct.fit_transform(X),dtype=np.float)

##encode the dependent variables usually two
#from sklearn.preprocessing import LabelEncoder
#le =LabelEncoder()
#Y=le.fit_transform(Y)


#Splitting into train and test set
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(
#        X,y, test_size=0.20, random state=0)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#
##feature_scaling
#from sklearn.preprocessing import StandardScaler
#sc_X=StandardScaler()
#
#X_train=sc_X.fit_transform(X_train)
#
#X_test=sc_X.fit_transform(X_test)



dataset1 = pd.read_csv("Salaries.csv.csv")


X= dataset1.iloc[:,0].values
Y=dataset1.iloc[:,1].values

#dataset1.describe()

#Transformers for missing value imputation


#Encoding Categorical Data]
#
#
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#
#ct = ColumnTransformer(transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[0])],remainder='passthrough')
##next fitting our data to the object
#
#X=np.array(ct.fit_transform(X),dtype=np.float)

##encode the dependent variables usually two
#from sklearn.preprocessing import LabelEncoder
#le =LabelEncoder()
#Y=le.fit_transform(Y)


#Splitting into train and test set
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(
#        X,y, test_size=0.20, random state=0)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
X_train= X_train.reshape(-1,1)
y_train= y_train.reshape(-1,1)
regressor.fit(X_train,y_train) 

#to reshape your data

#predicting the results
X_test = X_test.reshape(-1,1)

y_pred = regressor.predict(X_test)

#Visualization
plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Salary vs Experience(Training set results)')
plt.ylabel('Salary')
plt.show()

#vissualization
plt.scatter(X_test,y_test)
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Salary vs Experience(Test set results)')
plt.ylabel('Salary')
plt.show()


#Multiple Linear Regression


dataset2= pd.read_csv('Org_data.csv.csv')

X= dataset2.iloc[:,:-1].values
Y=dataset2.iloc[:,4].values



from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[3])],remainder='passthrough')
#next fitting our data to the object

X=np.array(ct.fit_transform(X),dtype=np.float)


#Avoid dummy variable trap
X=X[:,1:]



from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#use multiple linear regressio
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

#to reshape your data

#predicting the results
X_test = X_test.reshape(-1,1)

y_pred = regressor.predict(X_test)


#improving model performance
#backward eleimantion level
#need of add columns of ones
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int), values=X,axis=1)

#to fit model with all possible predictors
#1create a matrix of features 
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()


X_opt = X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()


X_opt = X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()


#POLYNOMIAL REGRESSION
# Data Pre-processing

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Gaming_data.csv.csv")
X = dataset.iloc[:,0:1].values
Y = dataset.iloc[:,1].values

#firring to linear
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#firring to polynomial regression
from sklearn.preprocessing import PolynomialFeatures
#to creare new matrix of features
poly_reg = PolynomialFeatures(degree=4) 

X_poly= poly_reg.fit_transform(X)

lin_reg2= LinearRegression()
lin_reg2.fit(X_poly,Y)

#Visualizing Linear regression result

plt.scatter(X,Y)
plt.plot(X,lin_reg.predict(X),color='red')
plt.title('Gaming data(Linear Regression)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()


plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='red')
plt.title('Gaming data(Linear Regression)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()

#predicting results
lin_reg.predict([[7.5]])
lin_reg2.predict(poly_reg.fit_transform([[11]]))


#ReGRESSION SVM



# Data Pre-processing

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Gaming_data.csv.csv")
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1:2].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y= sc_Y.fit_transform(Y)



#fitting SVR to datasetzz
from sklearn.svm import SVR
regressor = SVR(kernel ='rbf')
regressor.fit(X,Y.ravel())



#Visualising results

plt.scatter(X,Y)
plt.plot(X,regressor.predict(X),color='red')
plt.title('Gaming data(SVR)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()


#predicting results
y_pred = regressor.predict(sc_X.transform([[7.5]]))
y_pred = sc_Y.inverse_transform(y_pred)




#decision tree

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Gaming_data.csv.csv")
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1:2].values




#fitting decision tree regression to datasetzz
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)




#Visualising results of the decision tree

plt.scatter(X,Y)
plt.plot(X,regressor.predict(X),color='red')
plt.title('Gaming data(Decision Tree)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()


#predicting results
y_pred = regressor.predict(([[8.5]]))



#to view in higher resolution
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,Y)
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.title('Gaming data(Decision Tree)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()






#random forest

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Gaming_data.csv.csv")
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1:2].values




#fitting random forest regression to datasetzz
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200,random_state=0)
regressor.fit(X,Y.ravel())




#Visualising results of the decision tree

plt.scatter(X,Y)
plt.plot(X,regressor.predict(X),color='red')
plt.title('Gaming data(random forest')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()


#predicting results
y_pred = regressor.predict(([[8.5]]))



#to view in higher resolution
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,Y)
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.title('Gaming data(random forest)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()




#EVALUATING MODEL PERFORMANCE R SQUARED
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

dataset2= pd.read_csv('Org_data.csv.csv')

X= dataset2.iloc[:,:-1].values
Y=dataset2.iloc[:,4].values



from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[3])],remainder='passthrough')
#next fitting our data to the object

X=np.array(ct.fit_transform(X),dtype=np.float)


#Avoid dummy variable trap
X=X[:,1:]



from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#use multiple linear regressio
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

#to reshape your data

#predicting the results
X_test = X_test.reshape(-1,1)

y_pred = regressor.predict(X_test)


#improving model performance
#backward eleimantion level
#need of add columns of ones
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int), values=X,axis=1)

#to fit model with all possible predictors
#1create a matrix of features 
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()


X_opt = X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()


X_opt = X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

#to get the p-values and take the highest
regressor_OLS.summary()






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#CLAASSIFICATION ALGORIGTHM
# LOGISTIC REGRESSION PLUS DIMENSIONALITY REDUCTION 

# Data Pre-processing

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Customer List.csv.csv")
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:, 4].values


# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# PCA for dimensionality reduction
from sklearn.decomposition import PCA 
pca = PCA(n_components=None) #keep all components 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


explained_variance =pca.explained_variance_ratio_ #addition of these columns tells you number to keep i.e PCA = n_components =number oveer 50 %

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0,solver ='lbfgs')#add multi_class = 'multinomial' for PCA as a solver 
classifier.fit(X_train,Y_train)


y_pred=classifier.predict(X_test)


# var_prob=classifier.predict_proba(X_test)

#to see individual probabilities
# var_prob[0,:]

#build confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)






# LOGISTIC REGRESSION PLUS DIMENSIONALITY REDUCTION 
# LINEAR DISCRIMINANT ANALYSIS
# Data Pre-processing

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Customer List.csv.csv")
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:, 4].values


# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# PCA for dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

lda = lda(n_components=None) #keep all components 
X_train = lda.fit_transform(X_train,Y_train)
X_test = lda.transform(X_test)

# to determine the number of components to keep
explained_variance =lda.explained_variance_ratio_ #addition of these columns tells you number to keep i.e PCA = n_components =number oveer 50 %

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0,solver ='lbfgs')#add multi_class = 'multinomial' for PCA as a solver 
classifier.fit(X_train,Y_train)


y_pred=classifier.predict(X_test)


# var_prob=classifier.predict_proba(X_test)

#to see individual probabilities
# var_prob[0,:]

#build confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)




#KNN

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Customer List.csv.csv")
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:, 4].values


# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)



y_pred=classifier.predict(X_test)


var_prob=classifier.predict_proba(X_test)

#to see individual probabilities
var_prob[0,:]

#build confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)



#SVM


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Customer List.csv.csv")
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:, 4].values


# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier=SVC(kernel='linear', random_state=0)
classifier.fit(X_train,Y_train)



y_pred=classifier.predict(X_test)


#build confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)


#nAIVE BAYES



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Customer List.csv.csv")
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:, 4].values


# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train,Y_train)



y_pred=classifier.predict(X_test)


#build confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)

#classificationn decison tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Customer List.csv.csv")
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:, 4].values


# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)



from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)


y_pred=classifier.predict(X_test)


#build confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)

#Random Forrest Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Customer List.csv.csv")
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:, 4].values


# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)


y_pred=classifier.predict(X_test)


#build confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)



#K Means Cluster Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Shopping_center.csv.csv')


X=dataset.iloc[:,[3,4]].values

#Elbow method
from sklearn.cluster import KMeans
wcss=[]

#using 10 clusters
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++', n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()




#1fitting k_means to datasr=et
kmeans=KMeans(n_clusters=5,init='k-means++', n_init=10,max_iter=300,random_state=0)
Y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],s=100,c='cyan',label='Cluster 2')
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],s=100,c='blue',label='Cluster 4')
plt.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],s=100,c='magenta',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label ='Centroids')



plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Points')
plt.legend()
plt.show()






# Hierrarchical Cluster Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#doingn data preprocessing
dataset=pd.read_csv('Shopping_center.csv.csv')


X=dataset.iloc[:,[3,4]].values

#using dendrogram method

import scipy.cluster.hierarchy as sch
d_gram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('dendogram method')
plt.xlabel('customers')
plt.ylabel('Euclidean Distances')
plt.savefig('fig_1.png',dpi=500)
plt.show()


#fitting the HCA to dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
Y_hc = hc.fit_predict(X)


plt.scatter(X[Y_hc==0,0],X[Y_hc==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[Y_hc==1,0],X[Y_hc==1,1],s=100,c='cyan',label='Cluster 2')
plt.scatter(X[Y_hc==2,0],X[Y_hc==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[Y_hc==3,0],X[Y_hc==3,1],s=100,c='blue',label='Cluster 4')
plt.scatter(X[Y_hc==4,0],X[Y_hc==4,1],s=100,c='magenta',label='Cluster 5')


plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Points')
plt.legend()
plt.show()
plt.savefig('Hierachial_Cluster.png',dpi=500)


