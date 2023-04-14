
# from google.colab import drive
# drive.mount('/content/drive')

import streamlit as st 

import numpy as np #used to make numpy arrays
import pandas as pd #used to make data frame(structured table)
import matplotlib.pyplot as plt #plots and graph
import seaborn as sns #plots and graphs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics #evaluation purposes
import pandas as pd

df = pd.read_csv('D:/Py/heart_disease_prediction/heart1.csv')
df.head()
st.title("Heart Disease Data Analysis")
st.write("Data Set Size:",df.shape)
st.write("Dataset Shape:",df.shape)
print(df.columns)
print("Dataset Description:",df.describe())

# %%
print(df.isnull().sum())

# %%
print(df.info())
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True, cmap='terrain')

# %%
sns.pairplot(data=df)

# %%
df.hist(figsize=(12,12), layout=(5,3));



# %%
sns.catplot(data=df, x='sex', y='age',  hue='target', palette='husl')

# %%
sns.barplot(data=df, x='sex', y='chol', hue='target', palette='spring')

# %%
print(df['sex'].value_counts())

print(df['target'].value_counts())
print(df['thal'].value_counts())

# %% [markdown]
# <b>results of thallium stress test measuring blood flow to the heart, with possible values normal, fixed_defect, reversible_defect</b>

# %%
sns.countplot(x='sex', data=df, palette='husl', hue='target')
sns.countplot(x='target',palette='BuGn', data=df)
sns.countplot(x='ca',hue='target',data=df)

print(df['ca'].value_counts())
sns.countplot(x='thal',data=df, hue='target', palette='BuPu' )

sns.countplot(x='thal', hue='sex',data=df, palette='terrain')

print(df['cp'].value_counts())  # chest pain type

sns.countplot(x='cp' ,hue='target', data=df, palette='rocket')
sns.countplot(x='cp', hue='sex',data=df, palette='BrBG')

sns.boxplot(x='sex', y='chol', hue='target', palette='seismic', data=df)
sns.barplot(x='sex', y='cp', hue='target',data=df, palette='cividis')

sns.barplot(x='sex', y='thal', data=df, hue='target', palette='nipy_spectral')

sns.barplot(x='target', y='ca', hue='sex', data=df, palette='mako')

sns.barplot(x='sex', y='oldpeak', hue='target', palette='rainbow', data=df)


print(df['fbs'].value_counts())

# %%
sns.barplot(x='fbs', y='chol', hue='target', data=df,palette='plasma' )

# %%
sns.barplot(x='sex',y='target', hue='fbs',data=df)

# %% [markdown]
# ## **Cross Tables**
# 

# %%
gen = pd.crosstab(df['sex'], df['target'])
print(gen)

# %%
gen.plot(kind='bar', stacked=True, color=['green','yellow'], grid=False)

# %%
temp=pd.crosstab(index=df['sex'],
            columns=[df['thal']], 
            margins=True)
temp

# %%
temp.plot(kind="bar",stacked=True)
plt.show()

# %%
temp=pd.crosstab(index=df['target'],
            columns=[df['thal']], 
            margins=True)
temp

# %%
temp.plot(kind='bar', stacked=True)
plt.show()

# %%
chest_pain = pd.crosstab(df['cp'], df['target'])
chest_pain

# %%
chest_pain.plot(kind='bar', stacked=True, color=['purple','blue'], grid=False)

# %% [markdown]
# # **Preparing the data for Model**

# %% [markdown]
# ### **Scaling the data**

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()  
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])

# %%
df.head()

# %%
X= df.drop(['target'], axis=1)
y= df['target']

# %%
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=60)


# %% [markdown]
# ## **Check the sample Size**

# %%
print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

# %% [markdown]
# ## **Decision Tree**

# %%
  

# %%
from sklearn.tree import DecisionTreeClassifier


tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

# %%


# %%
test_score = accuracy_score(y_test, tree_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, tree_clf.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Decision Tree Clasifier", test_score]], 
                          columns=['Model','Testing Accuracy %'])
results_df

# %% [markdown]
# ## **K-NEAREST NEIGHBORING**

# %%
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)

# %%
test_score = accuracy_score(y_test, knn_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, knn_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["K-nearest neighbors", test_score]], 
                          columns=['Model','Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %% [markdown]
# ## **RANDOM FOREST SEARCH**

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

# %%
test_score = accuracy_score(y_test, rf_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, rf_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Random Forest Classifier", test_score]], 
                          columns=['Model','Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

# %%
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
df = pd.read_csv("D:/Py/heart_disease_prediction/heart1.csv")
X = df.iloc[:,0:20]  #independent columns
y = df.iloc[:,-1]    #target column
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(1,'Score'))

# %% [markdown]
# <font color = 'PURPLE'> <font size = 5><b>THIS IMPLIES THAT THE THALACH IS THE ATTRIBUTE THAT HELPS TO DETERMINE THE TARGET MORE THAN OTHER ATTRIBUTES</font></b>

# %%
import pandas as pd
import numpy as np
df = pd.read_csv("D:/Py/heart_disease_prediction/heart1.csv")
X = df.iloc[:,0:20]  #independent columns
y = df.iloc[:,-1]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# %%
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("D:/Py/heart_disease_prediction/heart1.csv")
X = df.iloc[:,0:20]  #independent columns
y = df.iloc[:,-1]    #target column i.e price range
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# %% [markdown]
# <font color = 'PURPLE'> <font size = 5><b>EVALUATION THE MODEL</font></b>

# %% [markdown]
# DTC

# %%
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()
model1=dtc.fit(X_train,y_train)
prediction1=model1.predict(X_test)
cm1= confusion_matrix(y_test,prediction1)

# %%
cm1

# %%
accuracy_score(y_test,prediction1)

# %%
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction1))
print('MSE:', metrics.mean_squared_error(y_test, prediction1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction1)))

# %%
from yellowbrick.model_selection import ValidationCurve
from sklearn.tree import DecisionTreeRegressor
#drawing the validation curve
val_curv = ValidationCurve(
    DecisionTreeRegressor(), param_name="max_depth",
    param_range=np.arange(1, 11), cv=10, scoring="r2"
)

# Fit and show the visualizer
val_curv.fit(X_test,y_test)
val_curv.show()

# %% [markdown]
# KNN

import joblib

# %%
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
model2 = KNN.fit(X_train, y_train)


prediction2 = model2.predict(X_test)
cm2= confusion_matrix(y_test, prediction2)
cm2

# %%
accuracy_score(y_test, prediction2)

# %%
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction2))
print('MSE:', metrics.mean_squared_error(y_test, prediction2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction2)))

# %%


# %%
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold 
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('RFS',RandomForestClassifier()))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Comparison between KNN , DTC and RFS')
ax = fig.add_subplot(111)
plt.boxplot(results)


plt.xlabel("ALGORITHMS")
plt.ylabel("ACCURACY")
ax.set_xticklabels(names)
plt.show()


# %%




joblib.dump(model2, 'D:/Py/heart_disease_prediction/model.pkl')
model3 = joblib.load('D:/Py/heart_disease_prediction/model.pkl')

print(X_test[0:5])

p = model3.predict(np.array((1,1,1,1,1,1,1,1,1,1,1,1,1)).reshape(-1,1))
print("streamlit predictions",p)


p = model3.predict(np.array(1,1,1,1,1,1,1,1,1,1,1,1,1).reshape(1,-1))
print("streamlit predictions",p)
