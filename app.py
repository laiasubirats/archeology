#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import datetime
 
# App config.
DEBUG = True
app = Flask(__name__,template_folder='static')
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
 
class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])
    @app.route("/", methods=['GET', 'POST'])
    def hello():
        form = ReusableForm(request.form)
        print(form.errors)
        if request.method == 'POST':
            name=request.form['name']
            surname=request.form['surname']
           
            
            i=1
            unarray=np.zeros(72)
            while i < 73:
                if (request.form['at'+str(i)] not in ''):
                    unarray[i-1]=float(request.form['at'+str(i)])
                i=i+1
            print(name)
            print(surname)
            toprint=pd.DataFrame(data=unarray)
            toprint=toprint.append([name])
            toprint=toprint.append([surname])
            toprint=toprint.append([datetime.datetime.now()])
            i=1
            while i<73:
                print(unarray[i-1])
                i=i+1 
            unarray = unarray.reshape(1, -1)
            prediction=clf.predict(unarray)
            print("Prediction: ",prediction[0])
            finalprediction=['Bovidae','Cervidae','Equidae','Leporidae', 'unknown']
            print(prediction)
            toprint=toprint.append([finalprediction[prediction[0]]])
            import os
            import sys
            #file_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
            #file_path = os.path.join(file_dir,'Students', str(datetime.datetime.now())+'.csv')
            #file_path = os.path.join("Documents/archeology/Students/",str(datetime.datetime.now()))
            #file_path=file_path.replace(" ","_").replace(":","_").replace("-","_").replace(".","_")
            #file_path="C://Users/laia.subirats/"+file_path+".csv"
            #print(file_path)
            #resultCSV=toprint.to_csv(file_path,index=False)
            
            surname=surname +". Your prediction is: "+str(finalprediction[prediction[0]])+ ".\n" 
            
        if form.validate():
    # Save the comment here.
            flash('Thank you ' + name +" "+ surname)
        else:
            flash('Warning: The name and surname form fields are required. ')
        return render_template('hello.html', form=form)
 
if __name__ == "__main__":
print("Inicio")
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import itertools


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import scipy.io as sio

param_grid_tree = {"max_depth": [3, None],
                   "max_features": [1, 3, 10],
                   "min_samples_split": [2, 3, 10],
                   "min_samples_leaf": [1, 3, 10],
                   "bootstrap": [True, False],
                   "criterion": ["gini", "entropy"]}

    param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

    

# Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
# Prepare the dataset
data_frame = pd.read_excel('data.xlsx')
#SC
import numpy as np
data_frame = data_frame.replace('-', np.NaN)
print("******",data_frame.columns)
#data_frame = data_frame.drop(['Fecha', 'Registro', 'Registro Diente', 'Dientes'], axis=1)
data_frame = data_frame.drop(['Fecha', 'Registro'], axis=1)

columns_name=["Hueso", "Grupo anatomico", "Color 1", "Color 2"]
for column in columns_name:
    print(data_frame[column])
    dict_column_aux = dict(enumerate(data_frame[column].unique()))
    print(dict_column_aux)
    dict_column = dict([(v, k) for k, v in dict_column_aux.items()])
    data_frame[column].replace(dict_column, inplace=True)

# Dataset just with numerical columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
data_frame_num = data_frame.select_dtypes(include=numerics)
data_frame_num = data_frame_num.fillna(0)  # Replace Nan values with zero
#print(data_frame_num.columns)  # Debug to check the numeric columns

# Change class labels for classes with less than 100 members to 'unknown'
familias = ['Anatidae', 'Bufonidae', 'Canidae', 'Corvidae', 'Erinaceidae', 'Felidae',
            'Phasianidae', 'Rhinocerotidae', 'Suidae', 'Testudinidae', 'Ursidae']
artiodactyla = ['Bovidae', 'Cervidae', 'Equidae']
labels_familia = (data_frame[data_frame['Familia'].notnull()].Familia).to_numpy()
# print(np.unique(label_familia_matrix), np.unique(labels_familia))
mask_unknown = np.isin(labels_familia, familias)
labels_unknow = labels_familia[mask_unknown]
labels_familia[mask_unknown] = 'unknown'
mask_artiodactyla = np.isin(labels_familia, artiodactyla)
labels_artiodactyla = labels_familia[mask_artiodactyla]

# Generate label matrix
from sklearn import preprocessing
le_familia = preprocessing.LabelEncoder()
label_familia_matrix = le_familia.fit_transform(labels_familia)
le_artiodactyla = preprocessing.LabelEncoder()
label_artiodactyla_matrix = le_artiodactyla.fit_transform(labels_artiodactyla)
# Debug purpose
import collections
print(sorted(collections.Counter(labels_familia).items()))
print(np.unique(label_familia_matrix), np.unique(labels_familia))

# Generate data matrix
train_val_familia_df = data_frame_num[data_frame['Familia'].notnull()]
train_val_familia_matrix = train_val_familia_df.to_numpy()
# Generate data matrix for the second classifier
train_val_artiodactyla_df = train_val_familia_df[mask_artiodactyla]
train_val_artiodactyla_matrix = train_val_artiodactyla_df.to_numpy()
print(np.shape(train_val_artiodactyla_matrix))

# Generate train and validation dataset for the main classifier
from sklearn.model_selection import train_test_split
X_train_familia_aux, X_test_familia_aux, y_train_familia_aux, y_test_familia_aux = train_test_split(train_val_familia_matrix,
                                                                                                label_familia_matrix,
                                                                                                test_size=0.33,
                                                                                                random_state=42)
# Generate train and validation dataset for the second classifier
X_train_artiodactyla_aux, X_test_artiodactyla_aux, y_train_artiodactyla_aux, y_test_artiodactyla_aux = train_test_split(train_val_artiodactyla_matrix,
                                                                                                label_artiodactyla_matrix,
                                                                                                test_size=0.33,
                                                                                                random_state=42)

#Random Forest
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn import preprocessing

#rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=2, oob_score = True) 

rf_pipeline = Pipeline([
                        ('std', preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)), 
                        ('smote', SMOTE(n_jobs=-1, random_state=42)),
                        ('classifier', RF())
                       ])
#pipeline = Pipeline([
#     ('smote', SMOTE(n_jobs=-1, random_state=42,kind='regular')),
#     ('normal',StandardScaler()),
#     ('clf',RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=2, oob_score = True))
#])

# Set the parameters by cross-validation
#'min_samples_leaf': [1,5,10,50,100,200,500],

#tuned_parameters = {
#    'classifier__n_estimators': [500],
#    'classifier__max_features': ['auto'],
#    'classifier__class_weight':[None]
#}
#{'classifier__max_features': 'auto', 'classifier__class_weight': None, 'classifier__n_estimators': 700
tuned_parameters = {  
    }  

#scores = ['precision', 'recall', 'f1']
#scores = [ 'f1_weighted','f1_micro','f1_macro','accuracy','roc_auc']
scores=['f1_macro']
from sklearn.naive_bayes import GaussianNB
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(GaussianNB(), tuned_parameters, cv=10,scoring='%s' % score)
    #clf = GridSearchCV(estimator=rfc, param_grid=tuned_parameters, cv=5,scoring='%s_weighted' % score,n_jobs= -1)
    #clf = GridSearchCV(estimator=rf_pipeline , param_grid=tuned_parameters, cv=10,scoring='%s' % score,n_jobs= -1)
    clf.fit(X_train_familia_aux, y_train_familia_aux)    
print("Finaaaaaal")  
app.run()  
#app.run(debug=False)
#app.run(host= '172.0.0.1', port=5000, debug=False)


# In[ ]:




