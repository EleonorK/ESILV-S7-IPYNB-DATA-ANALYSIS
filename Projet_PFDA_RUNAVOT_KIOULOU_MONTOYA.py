#!/usr/bin/env python
# coding: utf-8

# ### PROJET

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# # Data exportation and cleaning

# In[3]:


df = pd.read_csv("biodeg.csv",header=None,sep=";")
df


# In[4]:


col_names=[
"1) SpMax_L: Leading eigenvalue from Laplace matrix",
"2) J_Dz(e): Balaban-like index from Barysz matrix weighted by Sanderson electronegativity",
"3) nHM: Number of heavy atoms",
"4) F01[N-N]: Frequency of N-N at topological distance 1",
"5) F04[C-N]: Frequency of C-N at topological distance 4",
'6) NssssC: Number of atoms of type ssssC',
'7) nCb-: Number of substituted benzene C(sp2)',
'8) C%: Percentage of C atoms',
'9) nCp: Number of terminal primary C(sp3)',
'10) nO: Number of oxygen atoms',
'11) F03[C-N]: Frequency of C-N at topological distance 3',
'12) SdssC: Sum of dssC E-states',
'13) HyWi_B(m): Hyper-Wiener-like index (log function) from Burden matrix weighted by mass',
'14) LOC: Lopping centric index',
'15) SM6_L: Spectral moment of order 6 from Laplace matrix',
'16) F03[C-O]: Frequency of C - O at topological distance 3',
'17) Me: Mean atomic Sanderson electronegativity (scaled on Carbon atom)',
'18) Mi: Mean first ionization potential (scaled on Carbon atom)',
'19) nN-N: Number of N hydrazines',
'20) nArNO2: Number of nitro groups (aromatic)',
'21) nCRX3: Number of CRX3',
'22) SpPosA_B(p): Normalized spectral positive sum from Burden matrix weighted by polarizability',
'23) nCIR: Number of circuits',
'24) B01[C-Br]: Presence/absence of C - Br at topological distance 1',
'25) B03[C-Cl]: Presence/absence of C - Cl at topological distance 3',
'26) N-073: Ar2NH / Ar3N / Ar2N-Al / R..N..R',
'27) SpMax_A: Leading eigenvalue from adjacency matrix (Lovasz-Pelikan index)',
'28) Psi_i_1d: Intrinsic state pseudoconnectivity index - type 1d',
'29) B04[C-Br]: Presence/absence of C - Br at topological distance 4',
'30) SdO: Sum of dO E-states',
'31) TI2_L: Second Mohar index from Laplace matrix',
'32) nCrt: Number of ring tertiary C(sp3)',
'33) C-026: R--CX--R',
'34) F02[C-N]: Frequency of C - N at topological distance 2',
'35) nHDon: Number of donor atoms for H-bonds (N and O)',
'36) SpMax_B(m): Leading eigenvalue from Burden matrix weighted by mass',
'37) Psi_i_A: Intrinsic state pseudoconnectivity index - type S average',
'38) nN: Number of Nitrogen atoms',
'39) SM6_B(m): Spectral moment of order 6 from Burden matrix weighted by mass',
'40) nArCOOR: Number of esters (aromatic)',
'41) nX: Number of halogen atoms',
'42) experimental class: ready biodegradable (RB) and not ready biodegradable (NRB)']


# In[5]:


#Reducing column's names
col_acr=[]
for a in col_names:
    col_acr.append(re.findall('(?<=\) )(.*?)(?=\:)',a))
col_acr


# In[6]:


new_col_acr=[]
for a in col_acr:
    for e in a:
        new_col_acr.append(e)
new_col_acr     


# In[7]:


df.rename(columns=dict(zip(np.arange(0,42,1),new_col_acr)),inplace=True)


# In[8]:


pd.set_option('display.max_columns',df.shape[1])
df


# Get infos from data

# In[9]:


df.info()


# In[10]:


df.describe()


# # Data visualisation

# In[11]:


sns.boxplot(df["experimental class"],df['C%'])
plt.show()


# In[12]:


sns.boxplot(df["experimental class"],df['SdO'])
plt.show()


# In[13]:


sns.boxplot(df["experimental class"],df['LOC'])
plt.show()


# In[14]:


sns.boxplot(df["experimental class"],df['nO'])
plt.show()


# # Data preprocessing

# Splitting data into train and test split

# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('experimental class',axis=1),df['experimental class'], test_size=0.2, random_state=42)


# Scaling the data

# In[16]:



from sklearn import preprocessing

scaler=preprocessing.StandardScaler().fit(X_train)
X_train_scale=scaler.transform(X_train)
X_test_scale=scaler.transform(X_test)


# # Training, testing and compare models

# ### Support vector machine classifier

# In[17]:


from sklearn import svm

#Fitting and testing the model
cl_svm=svm.SVC()
cl_svm.fit(X_train_scale,y_train)

#Training set accuracy
print(cl_svm.score(X_train_scale,y_train))
#Testing set accuracy
print(cl_svm.score(X_test_scale,y_test))


# In[18]:


from sklearn.model_selection import GridSearchCV

param_svc={'kernel':['linear','rbf','poly','sigmoid'],'coef0':[0.01,0.1,0.3,0.5],'C':[1,100,300,500],'gamma':('auto','scale'),'degree':[1,3,5,7]}


# In[19]:


#GrisSearch function in order to selecting the model's best parameters easily

def EasyGrid(param,model,X,y,score='accuracy',cv=5,verb=3):
    grid = GridSearchCV(model,param,scoring=score,cv=cv,verbose=verb).fit(X,y)
    print("Best params are :",grid.best_params_)
    print("Best estimators are:",grid.best_estimator_)
    print("Best score is:",grid.best_score_)


# In[16]:


EasyGrid(param_svc,cl_svm,X_train_scale,y_train)


# In[20]:


#Same model but with optimized parameters
cl_svm=svm.SVC(kernel='rbf',C=1,coef0=0.01,degree=1,gamma='auto')
cl_svm.fit(X_train_scale,y_train)

print(cl_svm.score(X_train_scale,y_train))
print(cl_svm.score(X_test_scale,y_test))


# Plotting confusion matrix

# In[21]:


from sklearn import metrics

actual = y_test
predicted = cl_svm.predict(X_test_scale)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title("SVM classifier confusion matrix")
plt.show()


# ###  k-nearest neighbors classifier

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve


# In[23]:


model=KNeighborsClassifier()
k=np.arange(1,50)
train_scores,val_scores=validation_curve(model,X_train_scale,y_train,param_name='n_neighbors',param_range=k,cv=5)
plt.plot(k,val_scores.mean(axis=1),label='validation')
plt.plot(k,train_scores.mean(axis=1),c='orange',label='train')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()


# In[24]:


cl_knn= KNeighborsClassifier(n_neighbors=20)
cl_knn.fit(X_train_scale,y_train)

print(cl_knn.score(X_train_scale,y_train))
print(cl_knn.score(X_test_scale,y_test))


# ### Decision tree classifier

# In[25]:


from sklearn.tree import DecisionTreeClassifier

cl_dt= DecisionTreeClassifier()
cl_dt.fit(X_train_scale,y_train)

print(cl_dt.score(X_train_scale,y_train))
print(cl_dt.score(X_test_scale,y_test))


# In[56]:


param_dt={'criterion':['gini', 'entropy'],'splitter':['best', 'random'],'max_depth':np.arange(3,50,3)}
EasyGrid(param_dt,cl_dt, X_train_scale,y_train)


# In[26]:


cl_dt= DecisionTreeClassifier(criterion='entropy', max_depth=27, splitter='random')
cl_dt.fit(X_train_scale,y_train)

print(cl_dt.score(X_train_scale,y_train))
print(cl_dt.score(X_test_scale,y_test))


# Results are worst than default parameters, and we can see a huge overfitting of the model

# In[27]:


model=DecisionTreeClassifier()
k=np.arange(1,50)
train_scores,val_scores=validation_curve(model,X_train_scale,y_train,param_name='max_depth',param_range=k,cv=5)
plt.plot(k,val_scores.mean(axis=1),label='validation')
plt.plot(k,train_scores.mean(axis=1),c='orange',label='train')
plt.xlabel('max_depth')
plt.legend()
plt.show()


# In[28]:


cl_dt= DecisionTreeClassifier(criterion='entropy', max_depth=5, splitter='random')
cl_dt.fit(X_train_scale,y_train)

print(cl_dt.score(X_train_scale,y_train))
print(cl_dt.score(X_test_scale,y_test))


# Way much better, better score at test set and no more overfitting

# ### Decision tree classifier with Adaboosting

# In[30]:


param_ada={ "base_estimator__max_depth": [5,15,30,60,100],
            "base_estimator__criterion": ['gini', 'entropy'],
            'base_estimator__splitter':['best', 'random'],
            "n_estimators": [1,10,30,80,100,150],
            'learning_rate':[0.01,0.1,1,5]
    
}
EasyGrid(param_ada,cl_ada,X_train_scale,y_train)


# In[30]:


from sklearn.ensemble import AdaBoostClassifier


cl_ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=5),n_estimators=30,learning_rate=0.01)
cl_ada.fit(X_train_scale,y_train)

print(cl_ada.score(X_train_scale,y_train))
print(cl_ada.score(X_test_scale,y_test))


# Not bad, we again have overfitting

# In[31]:


model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=5))

k=np.arange(1,75)

train_scores,val_scores=validation_curve(model,X_train_scale,y_train,param_name='n_estimators',param_range=k,cv=5)
plt.plot(k,val_scores.mean(axis=1),label='validation')
plt.plot(k,train_scores.mean(axis=1),c='orange',label='train')
plt.xlabel('n_estimators')
plt.legend()
plt.show()


# We can see that overfitting is still here

# In[32]:


actual = y_test
predicted = cl_ada.predict(X_test_scale)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title("AdaBoosting confusion matrix")
plt.show()


# ### Random forest classifier

# In[33]:


from sklearn.ensemble import RandomForestClassifier


cl_rf= RandomForestClassifier()
cl_rf.fit(X_train_scale,y_train)

print(cl_rf.score(X_train_scale,y_train))
print(cl_rf.score(X_test_scale,y_test))


# In[69]:



param_rf={'n_estimators': [10,20,30,60,80,100,150],'criterion':['gini', 'entropy'],'max_depth':[None,10,15,20,25,30,50,100]}
EasyGrid(param_rf,cl_rf, X_train_scale,y_train)


# In[34]:



cl_rf=RandomForestClassifier(criterion='entropy', max_depth=100, min_samples_leaf=2,
                       min_samples_split=5)
cl_rf.fit(X_train_scale,y_train)

print(cl_rf.score(X_train_scale,y_train))
print(cl_rf.score(X_test_scale,y_test))


# In[35]:


model=RandomForestClassifier()

k=np.arange(1,50)

train_scores,val_scores=validation_curve(model,X_train_scale,y_train,param_name='n_estimators',param_range=k,cv=5)
plt.plot(k,val_scores.mean(axis=1),label='validation')
plt.plot(k,train_scores.mean(axis=1),c='orange',label='train')
plt.xlabel('n_estimators')
plt.legend()
plt.show()


# In[36]:


model=RandomForestClassifier()

k=np.arange(1,50)

train_scores,val_scores=validation_curve(model,X_train_scale,y_train,param_name='max_depth',param_range=k,cv=5)
plt.plot(k,val_scores.mean(axis=1),label='validation')
plt.plot(k,train_scores.mean(axis=1),c='orange',label='train')
plt.xlabel('max_depth')
plt.legend()
plt.show()


# Again, we have overfitting, and default parameters are the best

# ## Model Selection

# In[42]:


from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB



# prepare models
models = []
models.append(('SVC', cl_svm))
models.append(('KNN', cl_knn))
models.append(('DT', cl_dt))
models.append(('ADA DT', cl_ada))
models.append(('RF', cl_rf))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:

 cv_results = model_selection.cross_val_score(model,X_train_scale,y_train, cv=10, scoring=scoring)
 results.append(cv_results)
 names.append(name)


fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Support Vector Machine classifier seems to be the best and the more regular model

# In[96]:


from sklearn.metrics import RocCurveDisplay



ax = plt.gca()

for names,model in models:
    
    RocCurveDisplay.from_estimator(model,X_test_scale,y_test, alpha=0.8).plot(ax=ax, alpha=0.8)   
    
plt.show()


# SVC have here the best ROC curve so it confirms his selection.

# ### Conclusion

# The best model seems to be the Support Vector Machine classifier so that will be the one we will choose.

# In[32]:


#Showing parameters importante by permutation

from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(cl_svm, X_train_scale, y_train)

sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(20,20))
plt.barh(df.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

