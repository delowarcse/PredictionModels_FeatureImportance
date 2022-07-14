#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Calculate Feature Importance

# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')

# Load and merge datasets # white = control; red = stroke; wine = data
stroke_data = pd.read_csv('Stroke/Injured Participants Data.csv', delim_whitespace=False)
control_data = pd.read_csv('Healthy Control Participants Data.csv', delim_whitespace=False)

# store wine type as an attribute
stroke_data['data_type'] = 'stroke'   
control_data['data_type'] = 'control'

# merge control and stroke data
datas = pd.concat([stroke_data, control_data])
#datas = datas.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare Training and Testing Datasets
features = datas.iloc[:,:-1]
feature_names = features.columns
class_labels = np.array(datas['data_type'])

X_data = datas.iloc[:,:-1]
y_label = datas.iloc[:,-1]

# Data Normalization
ss = StandardScaler().fit(X_data)
X = ss.transform(X_data)
le = LabelEncoder()
le.fit(y_label)
y = le.transform(y_label)


# In[2]:


feature_names


# In[3]:


from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

kfold = KFold(n_splits=10, random_state=42, shuffle=True)
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}


# In[6]:


# Model Interpretation
# View Feature importances
from sklearn.linear_model import LogisticRegression
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

kfold = KFold(n_splits=10, random_state=42, shuffle=True)

lr_feature_importance = []
for i, (train_lr,test_lr) in enumerate(kfold.split(X,y)):
    plt.figure()
    model_lr = LogisticRegression().fit(X[train_lr],y[train_lr])
    lr_interpretation = Interpretation(X[test_lr], feature_names=feature_names)
    lr_in_model = InMemoryModel(model_lr.predict_proba, examples=X[train_lr], target_names=model_lr.classes_)
    lr_fea = lr_interpretation.feature_importance.feature_importance(lr_in_model, ascending=False)
    #print(lr_fea)
    lr_feature_importance.append(lr_fea)
    #plots = lr_interpretation.feature_importance.plot_feature_importance(lr_in_model, ascending=False)
    #plt.xlabel('Relative Importance Score')
    #plt.ylabel('Score')
print(lr_feature_importance)    


# In[35]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

#train model
#model_dt = DecisionTreeClassifier(max_depth=4)
KF_dt = KFold(n_splits=10, shuffle=True)

output_dt = cross_validate(estimator=model_dt,X=X,y=y,cv=kfold,scoring=scoring)
#model_dt.fit(X,y)
for idx,(train_dt,test_dt) in enumerate(KF_dt.split(X,y)):
    model_dt = DecisionTreeClassifier(max_depth=4).fit(X[train_dt], y[train_dt])
    print("Features sorted by their score for estimator {}:".format(idx))
    #feature_importances = model_dt.feature_importances_
    feature_importances = pd.DataFrame(model_dt.feature_importances_,
                                       index = feature_names,
                                        columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)
    
mean_fea = feature_importances.mean()
std_fea = feature_importances.std()
print(mean_fea)
print(std_fea)


# In[41]:


# https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/
# Support Vector Machine
from sklearn.ensemble import RandomForestClassifier

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

kfold = KFold(n_splits=10, random_state=42, shuffle=True)
model_rf=RandomForestClassifier(n_estimators=50) 

output_rf = cross_validate(estimator=model_rf, X=X, y=y, cv=kfold, scoring=scoring, return_estimator=True)
for idx,estimator in enumerate(output_rf['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    feature_importances = pd.DataFrame(estimator.feature_importances_,
                                       index = feature_names,
                                        columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)


# In[32]:


# train the Random Forest Hyperparameter Tuning
from sklearn.ensemble import RandomForestClassifier

model_rft = RandomForestClassifier(n_estimators=200, max_features='auto')

results_rft = cross_validate(estimator=model_rft,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring,
                            return_estimator=True)
for i, estimator_rft in enumerate(results_rft['estimator']):
    print("Feature Sorted by their score for estimator {}".format(i))
    feature_importance_rft = pd.DataFrame(estimator_rft.feature_importances_,
                                         index=feature_names,
                                         columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importance_rft)


# In[45]:


from sklearn.svm import SVC
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

kf_svm = KFold(n_splits=10, shuffle=True)
for i, (train_svm,test_svm) in enumerate(kf_svm.split(X,y)):
    model_svm = SVC(probability=True).fit(X[train_svm], y[train_svm])
    svm_interpretation = Interpretation(X[test_svm], feature_names=feature_names)
    svm_in_memory = InMemoryModel(model_svm.predict_proba, examples=X[train_svm], target_names=model_svm.classes_)
    svm_feature_importance = svm_interpretation.feature_importance.feature_importance(svm_in_memory, ascending=False)
    print(svm_feature_importance)
    


# In[11]:


from sklearn.svm import SVC

model_svm = SVC(random_state=42)

results_svm = cross_validate(estimator=model_svm,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring)
print(results_svm)


# In[13]:


# https://chrisalbon.com/deep_learning/keras/k-fold_cross-validating_neural_networks/
from keras import models, layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def create_network():
    # network defination
    network = models.Sequential()
    # Add fully connected layer with a ReLU
    network.add(layers.Dense(units=16, activation='relu', input_shape=(12,)))
    network.add(layers.Dense(units=16, activation='relu'))
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # compile network
    network.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    return network


# In[14]:


# wrap keras model
neural_network = KerasClassifier(build_fn=create_network,
                                epochs=100,
                                batch_size=10,
                                verbose=1)


# In[15]:


results_dnn = cross_validate(estimator=neural_network,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring)
print(results_dnn)


# In[16]:


print('Deep Neural Network Metrics:')
print('Fit time:',results_dnn['fit_time'])
print('Score time:',results_dnn['score_time'])
print('10-fold Accuracy:',results_dnn['test_accuracy'])
print('Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_dnn['test_accuracy']),np.std(results_dnn['test_accuracy'])))
print('10-fold Precision:',results_dnn['test_precision'])
print('Precision(Mean (Standard Deviation): %f (%f)'%(np.mean(results_dnn['test_precision']),np.std(results_dnn['test_precision'])))
print('10-fold Recall:',results_dnn['test_recall'])
print('Recall(Mean (Standard Deviation): %f (%f)'%(np.mean(results_dnn['test_recall']),np.std(results_dnn['test_recall'])))
print('10-fold f1-score:',results_dnn['test_f1_score'])
print('f1-score(Mean (Standard Deviation): %f (%f)'%(np.mean(results_dnn['test_f1_score']),np.std(results_dnn['test_f1_score'])))


# In[17]:


print('Logistic Regression: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_lr['test_accuracy']),np.std(results_lr['test_accuracy'])))
print('Decision Tree: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_dt['test_accuracy']),np.std(results_dt['test_accuracy'])))
print('Random Forest: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_rf['test_accuracy']),np.std(results_rf['test_accuracy'])))
print('Random Forest with Hyperparameters Tuning: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_rft['test_accuracy']),np.std(results_rft['test_accuracy'])))
print('Support Vector Machine: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_svm['test_accuracy']),np.std(results_svm['test_accuracy'])))
print('Deep Neural Network: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_dnn['test_accuracy']),np.std(results_dnn['test_accuracy'])))


# In[ ]:




