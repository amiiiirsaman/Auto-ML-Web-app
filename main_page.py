#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import cufflinks
import warnings
import random
from random import shuffle
import functions

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer 
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier

from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD


cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
warnings.filterwarnings("ignore")


#configuration of the page

img = Image.open("AArete.png")
st.set_page_config(layout = "wide", page_icon = 'AArete.png', page_title='ML Web APP', initial_sidebar_state="expanded", 
     menu_items={'About': "https://www.aarete.com/our-solutions/digital-technology/data-analytics/"})

def get_colors():
        s='''
            aliceblue, antiquewhite, aqua, aquamarine, azure,
            beige, bisque, black, blanchedalmond, blue,
            blueviolet, brown, burlywood, cadetblue,
            chartreuse, chocolate, coral, cornflowerblue,
            cornsilk, crimson, cyan, darkblue, darkcyan,
            darkgoldenrod, darkgray, darkgrey, darkgreen,
            darkkhaki, darkmagenta, darkolivegreen, darkorange,
            darkorchid, darkred, darksalmon, darkseagreen,
            darkslateblue, darkslategray, darkslategrey,
            darkturquoise, darkviolet, deeppink, deepskyblue,
            dimgray, dimgrey, dodgerblue, firebrick,
            floralwhite, forestgreen, fuchsia, gainsboro,
            ghostwhite, gold, goldenrod, gray, grey, green,
            greenyellow, honeydew, hotpink, indianred, indigo,
            ivory, khaki, lavender, lavenderblush, lawngreen,
            lemonchiffon, lightblue, lightcoral, lightcyan,
            lightgoldenrodyellow, lightgray, lightgrey,
            lightgreen, lightpink, lightsalmon, lightseagreen,
            lightskyblue, lightslategray, lightslategrey,
            lightsteelblue, lightyellow, lime, limegreen,
            linen, magenta, maroon, mediumaquamarine,
            mediumblue, mediumorchid, mediumpurple,
            mediumseagreen, mediumslateblue, mediumspringgreen,
            mediumturquoise, mediumvioletred, midnightblue,
            mintcream, mistyrose, moccasin, navajowhite, navy,
            oldlace, olive, olivedrab, orange, orangered,
            orchid, palegoldenrod, palegreen, paleturquoise,
            palevioletred, papayawhip, peachpuff, peru, pink,
            plum, powderblue, purple, red, rosybrown,
            royalblue, saddlebrown, salmon, sandybrown,
            seagreen, seashell, sienna, silver, skyblue,
            slateblue, slategray, slategrey, snow, springgreen,
            steelblue, tan, teal, thistle, tomato, turquoise,
            violet, wheat, white, whitesmoke, yellow,
            yellowgreen
            '''
        li=s.split(',')
        li=[l.replace('\n','') for l in li]
        li=[l.replace(' ','') for l in li]
        shuffle(li)
        return li

colors = get_colors()   

#Loading the data
@st.cache
def get_data_classification():
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'heart_statlog.csv'))
    df.loc[df['chest pain type'] == 1, 'chest pain type'] = 'typical angina'
    df.loc[df['chest pain type'] == 2, 'chest pain type'] = 'atypical angina'
    df.loc[df['chest pain type'] == 3, 'chest pain type'] = 'non-anginal pain'
    df.loc[df['chest pain type'] == 4, 'chest pain type'] = 'asymptomatic'
    df['chest pain type'] = df['chest pain type'].astype(str)

    df.loc[df['sex'] == 1, 'sex'] = 'male'
    df.loc[df['sex'] == 0, 'sex'] = 'female'
    df['sex'] = df['sex'].astype(str)

    df.loc[df['resting ecg'] == 0, 'resting ecg'] = 'normal'
    df.loc[df['resting ecg'] == 1, 'resting ecg'] = 'ST-T wave abnormality'
    df.loc[df['resting ecg'] == 2, 'resting ecg'] = 'probable or definite left ventricular hypertrophy'
    df['resting ecg'] = df['resting ecg'].astype(str)

    df.loc[df['exercise angina'] == 0, 'exercise angina'] = 'no'
    df.loc[df['exercise angina'] == 1, 'exercise angina'] = 'yes'
    df['exercise angina'] = df['exercise angina'].astype(str)

    df.loc[df['ST slope'] == 0, 'ST slope'] = 'unsloping'
    df.loc[df['ST slope'] == 1, 'ST slope'] = 'flat'
    df.loc[df['ST slope'] == 2, 'ST slope'] = 'downslopping'
    df['ST slope'] = df['ST slope'].astype(str)
    return df


def get_imputer(imputer):
    if imputer == 'None':
        return 'drop'
    if imputer == 'Most frequent value':
        return SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    if imputer == 'Mean':
        return SimpleImputer(strategy='mean', missing_values=np.nan)
    if imputer == 'Median':
        return SimpleImputer(strategy='median', missing_values=np.nan)


def get_pipeline_missing_num(imputer, scaler):
    if imputer == 'None':
        return 'drop'
    if imputer == 'Mean':
        pipeline = make_pipeline(SimpleImputer(strategy='mean', missing_values=np.nan))
    if imputer == 'Median':
        pipeline = make_pipeline(SimpleImputer(strategy='median', missing_values=np.nan))
    if(scaler != 'None'):
        pipeline.steps.append(('scaling', get_scaling(scaler)))
    return pipeline


def get_pipeline_missing_cat(imputer, encoder):
    if imputer == 'None' or encoder == 'None':
        return 'drop'
    if imputer == 'Most frequent value':
        pipeline = make_pipeline(SimpleImputer(strategy='most_frequent', missing_values=np.nan))
    pipeline.steps.append(('encoding', get_encoding(encoder)))
    return pipeline


def get_encoding(encoder):
    if encoder == 'None':
        return 'drop'
    if encoder == 'Ordinal encoder':
        return OrdinalEncoder(handle_unknown='use_encoded_value')
    if encoder == 'OneHotEncoder':
        return OneHotEncoder(handle_unknown='ignore')
    if encoder == 'CountVectorizer':
        return CountVectorizer()
    if encoder == 'TfidfVectorizer':
        return TfidfVectorizer()



def get_scaling(scaler):
    if scaler == 'None':
        return 'passthrough'
    if scaler == 'Standard scaler':
        return StandardScaler()
    if scaler == 'MinMax scaler':
        return MinMaxScaler()
    if scaler == 'Robust scaler':
        return RobustScaler()


# In[12]:


def get_clf_ml_algorithm(algorithm, hyperparameters):
    if algorithm == 'Logistic Regression':
        return LogisticRegression(solver=hyperparameters['solver'])
    if algorithm == 'Support vector':
        return SVC(kernel = hyperparameters['kernel'], C = hyperparameters['C'])
    if algorithm == 'Naive Bayes':
        return GaussianNB()
    if algorithm == 'K Nearest Neighbors':
        return KNeighborsClassifier(n_neighbors = hyperparameters['n_neighbors'], metric = hyperparameters['metric'], weights = hyperparameters['weights'])
    if algorithm == 'Ridge Classifier':
        return RidgeClassifier(alpha=hyperparameters['alpha'], solver=hyperparameters['solver'])
    if algorithm == 'Decision Tree':
        return DecisionTreeClassifier(criterion = hyperparameters['criterion'], min_samples_split = hyperparameters['min_samples_split'])
    if algorithm == 'Random Forest':
        return RandomForestClassifier(n_estimators = hyperparameters['n_estimators'], criterion = hyperparameters['criterion'], min_samples_split = hyperparameters['min_samples_split'])
    if algorithm == 'XGBoost':
        return XGBClassifier(n_estimators = hyperparameters["n_estimators"], max_depth = hyperparameters["max_depth"], learning_rate = hyperparameters["Learning Rate"], objective = hyperparameters["Objective"], gamma = hyperparameters["Gamma"], reg_alpha = hyperparameters["reg_alpha"], reg_lambda = hyperparameters["reg_lambda"], colsample_bytree = hyperparameters["colsample_bytree"])

    

def get_reg_ml_algorithm(algorithm, hyperparameters):
    if algorithm == 'Linear Regression':
        return LinearRegression()

    if algorithm == 'SGD':
        return SGDRegressor(alpha = hyperparameters['alpha'], loss = hyperparameters['loss'], penalty = hyperparameters['penalty'], learning_rate = hyperparameters['learning_rate'], random_state = 42)
    
    if algorithm == 'K Nearest Neighbors':
        return KNeighborsRegressor(n_neighbors = hyperparameters['n_neighbors'], metric = hyperparameters['metric'], weights = hyperparameters['weights'])

    if algorithm == 'Decision Tree':
        return DecisionTreeRegressor(criterion = hyperparameters['criterion'], max_depth = hyperparameters['max_depth'], min_samples_leaf = hyperparameters['min_samples_leaf'], random_state = 42)
                                                                        
    if algorithm == 'Random Forest':
        return RandomForestRegressor(n_estimators = hyperparameters["n_estimators"], max_depth = hyperparameters['max_depth'], random_state = 42)
    
    if algorithm == 'XGBoost':
        return XGBRegressor(n_estimators = hyperparameters["n_estimators"], max_depth = hyperparameters["max_depth"], learning_rate = hyperparameters["Learning Rate"], gamma = hyperparameters["Gamma"], reg_alpha = hyperparameters["reg_alpha"], reg_lambda = hyperparameters["reg_lambda"], colsample_bytree = hyperparameters["colsample_bytree"], random_state = 42)

    if algorithm == 'Neural Network':
        return MLPRegressor(hidden_layer_sizes = hyperparameters['hidden_layer_sizes'], activation = hyperparameters['activation'], solver = hyperparameters['solver'], alpha = hyperparameters['alpha'], random_state = 42)
                                


def get_dim_reduc_algo(algorithm, hyperparameters):
    if algorithm == 'None':
        return 'passthrough'
    if algorithm == 'PCA':
        return PCA(n_components = hyperparameters['n_components'])
    if algorithm == 'LDA':
        return LDA(solver = hyperparameters['solver'])
    if algorithm == 'Kernel PCA':
        return KernelPCA(n_components = hyperparameters['n_components'], kernel = hyperparameters['kernel'])
    if algorithm == 'Truncated SVD':
        return TruncatedSVD(n_components = hyperparameters['n_components'])


def get_fold(algorithm, nb_splits):
    if algorithm == 'Kfold':
        return KFold(n_plits = nb_splits, shuffle=True, random_state = 0)
    if algorithm == 'StratifiedKFold':
        return StratifiedKFold()

SPACER = .2
ROW = 1

    
def main_page():
    st.markdown("# Main page ")
    st.sidebar.markdown("# Main page")
    st.markdown('***')

    title_spacer1, title, title_spacer_2 = st.columns((.1,ROW,.1))

    with title:
        st.title('Classification & Regression Modeling Tool')
        st.markdown("""
                This app allows you to test various machine learning algorithms and combinations of preprocessing techniques 
                on your desired dataset.

                * Use the menu on the left to select the ML algorithm and hyperparameters
                * Click on "how to use this app?" to learn more.
                """)

    title_spacer2, title_2, title_spacer_2 = st.columns((.1,ROW,.1))

    with title_2:
        with st.expander("How to use this app?"):
            st.markdown("""
                This app allows you to test different machine learning algorithms and combinations of preprocessing techniques.
                The menu on the left allows you to choose:
                * the columns to drop (either by % of missing value or by name)
                * the transfomation to apply on your columns (imputation, scaling, encoding...)
                * the dimension reduction algorithm (none, PCA, LDA, kernel PCA)
                * the type of cross validation (KFold, StratifiedKFold)
                * the machine learning algorithm and its hyperparameters
                """)
            st.write("")
            st.markdown("""
                Each time you modify a parameter, the algorithm applies the modifications and outputs the preprocessed dataset and the results of the cross validation.
            """)

def page2():
    st.markdown("# Classification Modeling")
    st.sidebar.markdown("# Classification Modeling")
    st.sidebar.markdown("****************")

    
    data = pd.read_csv('train_titanic.csv')
    target_selected = 'Survived'
   
    functions.space()
    st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)
    df = data
    file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
    dataset = st.file_uploader(label = '')

    
    use_defo = st.checkbox('Use example Dataset')
    if use_defo:
        dataset = 'train_titanic.csv'

    if dataset:
        if file_format == 'csv' or use_defo:
            df = pd.read_csv(dataset)
        else:
            df = pd.read_excel(dataset)    

    columns = df.columns.to_list()

    obj_colss = [col for col in df.select_dtypes(include=['object']).columns]
    num_colss = [col for col in df.select_dtypes(include='number').columns]

    cat_colss = []

    for col in num_colss:
        if(len(df[col].unique()) < 25):
            cat_colss.append(col)

    for col in obj_colss:
        if(len(df[col].unique()) < 25):
            cat_colss.append(col)


    target = st.sidebar.selectbox('Select Target Variable', cat_colss)
    Y = df[target].values.ravel()
    Z = df
    del Z[target]
    
    features = st.sidebar.multiselect("Select Features ", Z.columns.to_list(), default=Z.columns.to_list()[0:5])
    X = df[features]
    

    #Sidebar 
    #selection box for the different features
    st.sidebar.title('Preprocessing')
    st.sidebar.subheader('Dropping columns')
    missing_value_threshold_selected = st.sidebar.slider('Max missing values in feature (%)', 0,100,30,1)

    st.sidebar.subheader('Column Transformation')
    categorical_imputer_selected = st.sidebar.selectbox('Handling categorical missing values', ['None','Most frequent value'], index = 1)
    numerical_imputer_selected = st.sidebar.selectbox('Handling numerical missing values', ['None','Median', 'Mean'], index = 2)

    encoder_selected = st.sidebar.selectbox('Encoding categorical values', ['None', 'OneHotEncoder'], index = 1)
    scaler_selected = st.sidebar.selectbox('Scaling', ['None', 'Standard scaler', 'MinMax scaler', 'Robust scaler'], index = 2)
    text_encoder_selected = st.sidebar.selectbox('Encoding text values', ['None', 'CountVectorizer', 'TfidfVectorizer'])

    st.header('Original dataset')

    row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns((SPACER/10,ROW*1.5,SPACER,ROW, SPACER/10))

    with row1_1:
        st.write(df)

    with row1_2:
        number_features = len(X.columns)

        #feature with missing values
        drop_cols = []
        for col in X.columns:
            #put the feature in the drop trable if threshold not respected
            if((X[col].isna().sum()/len(X)*100 > missing_value_threshold_selected) & (col not in drop_cols)):
                drop_cols.append(col)


        #numerical columns
        num_cols_extracted = [col for col in X.select_dtypes(include='number').columns]
        num_cols = []
        num_cols_missing = []
        cat_cols = []
        cat_cols_missing = []
        for col in num_cols_extracted:
            if(len(X[col].unique()) < 25):
                cat_cols.append(col)
            else:
                num_cols.append(col)

        #categorical columns
        obj_cols = [col for col in X.select_dtypes(include=['object']).columns]
        text_cols = []
        text_cols_missing = []
        for col in obj_cols:
            if(len(X[col].unique()) < 25):
                cat_cols.append(col)
            else:
                text_cols.append(col)

        #remove dropped columns
        for element in drop_cols:
            if element in num_cols:
                num_cols.remove(element)
            if element in cat_cols:
                cat_cols.remove(element)
            if element in text_cols:
                text_cols.remove(element)

        #display info on dataset
        st.write('Original size of the dataset', X.shape)
        st.write('Dropping ', round(100*len(drop_cols)/number_features,2), '% of feature for missing values')
        st.write('Numerical columns : ', round(100*len(num_cols)/number_features,2), '%')
        st.write('Categorical columns : ', round(100*len(cat_cols)/number_features,2), '%')
        st.write('Text columns : ', round(100*len(text_cols)/number_features,2), '%')

        st.write('Total : ', round(100*(len(drop_cols)+len(num_cols)+len(cat_cols)+len(text_cols))/number_features,2), '%')

        #create new lists for columns with missing elements
        for col in X.columns:
            if (col in num_cols and X[col].isna().sum() > 0):
                num_cols.remove(col)
                num_cols_missing.append(col)
            if (col in cat_cols and X[col].isna().sum() > 0):
                cat_cols.remove(col)
                cat_cols_missing.append(col)
            # if (col in text_cols and X[col].isna().sum() > 0):
            #     text_cols.remove(col)
            #     text_cols_missing.append(col)

        #combine text columns in one new column because countVectorizer does not accept multiple columns
        X['text'] = X[text_cols].astype(str).agg(' '.join, axis=1)
        for cols in text_cols:
            drop_cols.append(cols)
        text_cols = 'text'


    #need to make two preprocessing pipeline too handle the case encoding without imputer...
    preprocessing = make_column_transformer(
        (get_pipeline_missing_cat(categorical_imputer_selected, encoder_selected) , cat_cols_missing),
        (get_pipeline_missing_num(numerical_imputer_selected, scaler_selected) , num_cols_missing),

        (get_encoding(encoder_selected), cat_cols),
        (get_encoding(text_encoder_selected), text_cols),
        (get_scaling(scaler_selected), num_cols)
    )


    dim = preprocessing.fit_transform(X).shape[1]
    if((encoder_selected == 'OneHotEncoder') | (dim > 2)):
        dim = dim - 1

    if (dim > 2):
        st.sidebar.title('Dimension Reduction')
        dimension_reduction_algorithm_selected = st.sidebar.selectbox('Algorithm', ['None', 'Kernel PCA'])

        hyperparameters_dim_reduc = {}                                      
        # if(dimension_reduction_algorithm_selected == 'PCA'):
        #     hyperparameters_dim_reduc['n_components'] = st.sidebar.slider('Number of components (default = nb of features - 1)', 2, dim, dim, 1)
        # if(dimension_reduction_algorithm_selected == 'LDA'):
        #     hyperparameters_dim_reduc['solver'] = st.sidebar.selectbox('Solver (default = svd)', ['svd', 'lsqr', 'eigen'])
        if(dimension_reduction_algorithm_selected == 'Kernel PCA'):
            hyperparameters_dim_reduc['n_components'] = st.sidebar.slider('Number of components (default = nb of features - 1)', 2, dim, dim, 1)
            hyperparameters_dim_reduc['kernel'] = st.sidebar.selectbox('Kernel (default = linear)', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
        # if(dimension_reduction_algorithm_selected == 'Truncated SVD'):
        #     hyperparameters_dim_reduc['n_components'] = st.sidebar.slider('Number of components (default = nb of features - 1)', 2, dim, dim, 1)
    else :
        st.sidebar.title('Dimension Reduction')
        dimension_reduction_algorithm_selected = st.sidebar.selectbox('Number of features too low', ['None'])
        hyperparameters_dim_reduc = {}         

    st.sidebar.title('Cross Validation')
    type = st.sidebar.selectbox('Type', ['KFold', 'StratifiedKFold'])
    nb_splits = st.sidebar.slider('Number of splits', min_value=3, max_value=20)
    folds = get_fold(type, nb_splits)

    st.sidebar.title('Model Selection')
    classifier_list = ['Logistic Regression', 'Support Vector', 'K Nearest Neighbors', 'Naive Bayes', 'Ridge Classifier', 'Decision Tree', 'Random Forest', 'XGBoost']
    classifier_selected = st.sidebar.selectbox('', classifier_list)

    st.sidebar.header('Hyperparameters selection')
    hyperparameters = {}

    if(classifier_selected == 'Logistic Regression'):
        hyperparameters['solver'] = st.sidebar.selectbox('Solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        if (hyperparameters['solver'] == 'liblinear'):
            hyperparameters['penalty'] = st.sidebar.selectbox('Penalty (default = l2)', ['none', 'l1', 'l2'])
        if (hyperparameters['solver'] == 'saga'):
            hyperparameters['penalty'] = st.sidebar.selectbox('Penalty (default = l2)', ['none', 'l1', 'l2', 'elasticnet'])
        else:
            hyperparameters['penalty'] = st.sidebar.selectbox('Penalty (default = l2)', ['none', 'l2'])
        hyperparameters['C'] = st.sidebar.selectbox('C (default = 1.0)', [100, 10, 1, 0.1, 0.01])

    if(classifier_selected == 'Ridge Classifier'):
        hyperparameters['alpha'] = st.sidebar.slider('Alpha (default value = 1.0)', 0.0, 10.0, 1.0, 0.1)
        hyperparameters['solver'] = st.sidebar.selectbox('Solver (default = auto)', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])

    if(classifier_selected == 'K Nearest Neighbors'):
        hyperparameters['n_neighbors'] = st.sidebar.slider('Number of neighbors (default value = 5)', 1, 21, 5, 1)
        hyperparameters['metric'] = st.sidebar.selectbox('Metric (default = minkowski)', ['minkowski', 'euclidean', 'manhattan', 'chebyshev'])
        hyperparameters['weights'] = st.sidebar.selectbox('Weights (default = uniform)', ['uniform', 'distance'])

    if(classifier_selected == 'Support Vector'):
        hyperparameters['kernel'] = st.sidebar.selectbox('Kernel (default = rbf)', ['rbf', 'linear', 'poly', 'sigmoid'])
        hyperparameters['C'] = st.sidebar.selectbox('C (default = 1.0)', [100, 10, 1, 0.1, 0.01])

    if(classifier_selected == 'Decision Tree'):
        hyperparameters['criterion'] = st.sidebar.selectbox('Criterion (default = gini)', ['gini', 'entropy'])
        hyperparameters['min_samples_split'] = st.sidebar.slider('Min sample splits (default = 2)', 2, 20, 2, 1)

    if(classifier_selected == 'Random Forest'):
        hyperparameters['n_estimators'] = st.sidebar.slider('Number of estimators (default = 100)', 10, 500, 100, 10)
        hyperparameters['criterion'] = st.sidebar.selectbox('Criterion (default = gini)', ['gini', 'entropy'])
        hyperparameters['min_samples_split'] = st.sidebar.slider('Min sample splits (default = 2)', 2, 20, 2, 1)

    if(classifier_selected == 'XGBoost'):
        hyperparameters['n_estimators'] = st.sidebar.slider("n_estimators", 50, 500, step=50, value=50)
        hyperparameters['Learning Rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.5,value=0.1)
        hyperparameters['Objective'] = st.sidebar.selectbox("Objective", ('binary:logistic','reg:logistic','reg:squarederror',"reg:gamma"))
        hyperparameters['max_depth'] = st.sidebar.slider("max_depth", 1, 20,value=6)
        hyperparameters['Gamma'] = st.sidebar.slider("Gamma",0,10,value=5)
        hyperparameters['reg_lambda'] = st.sidebar.slider("reg_lambda",1.0,5.0,step=0.1)
        hyperparameters['reg_alpha'] = st.sidebar.slider("reg_alpha",0.0,5.0,step=0.1)
        hyperparameters['colsample_bytree'] = st.sidebar.slider("colsample_bytree",0.5,1.0,step=0.1)

    # In[ ]:


    preprocessing_pipeline = Pipeline([
        ('preprocessing' , preprocessing),
        ('dimension reduction', get_dim_reduc_algo(dimension_reduction_algorithm_selected, hyperparameters_dim_reduc))
    ])


    pipeline = Pipeline([
        ('preprocessing' , preprocessing),
        ('dimension reduction', get_dim_reduc_algo(dimension_reduction_algorithm_selected, hyperparameters_dim_reduc)),
        ('ml', get_clf_ml_algorithm(classifier_selected, hyperparameters))
    ])

    cv_score = cross_val_score(pipeline, X, Y, cv=folds, scoring='accuracy')

    preprocessing_pipeline.fit(X)
    X_preprocessed = preprocessing_pipeline.transform(X)

    st.header('Preprocessed dataset')
    st.write(X_preprocessed)
    
    st.subheader('Results')
    st.write('Accuracy : ', round(cv_score.mean()*100,2), '%')
    st.write('Standard Deviation of Accuracy: ', round(cv_score.std()*100,2), '%')

def page3():
    st.markdown("# Regression Modeling")
    st.sidebar.markdown("# Regression Modeling")
    st.sidebar.markdown("****************")
    
    data = pd.read_csv('train_titanic.csv')
    target_selected = 'Fare'
   
    functions.space()
    st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)
    df = data
    file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
    dataset = st.file_uploader(label = '')

    
    use_defo = st.checkbox('Use example Dataset')
    if use_defo:
        dataset = 'train_titanic.csv'

    if dataset:
        if file_format == 'csv' or use_defo:
            df = pd.read_csv(dataset)
        else:
            df = pd.read_excel(dataset)    

    columns = df.columns.to_list()

    num_colss = [col for col in df.select_dtypes(include='number').columns]

    target = st.sidebar.selectbox('Select Target Variable', num_colss)
    Y = df[target].values.ravel()
    Z = df
    del Z[target]
    
    features = st.sidebar.multiselect("Select Features ", Z.columns.to_list(), default=Z.columns.to_list()[0:5])
    X = df[features]
    

    #Sidebar 
    #selection box for the different features
    st.sidebar.title('Preprocessing')
    st.sidebar.subheader('Dropping columns')
    missing_value_threshold_selected = st.sidebar.slider('Max missing values in feature (%)', 0,100,30,1)

    st.sidebar.subheader('Column Transformation')
    categorical_imputer_selected = st.sidebar.selectbox('Handling categorical missing values', ['None','Most frequent value'], index = 1)
    numerical_imputer_selected = st.sidebar.selectbox('Handling numerical missing values', ['None','Median', 'Mean'], index = 2)

    encoder_selected = st.sidebar.selectbox('Encoding categorical values', ['None', 'OneHotEncoder'], index = 1)
    scaler_selected = st.sidebar.selectbox('Scaling', ['None', 'Standard scaler', 'MinMax scaler', 'Robust scaler'], index = 2)
    text_encoder_selected = st.sidebar.selectbox('Encoding text values', ['None', 'CountVectorizer', 'TfidfVectorizer'])

    st.header('Original dataset')

    row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns((SPACER/10,ROW*1.5,SPACER,ROW, SPACER/10))

    with row1_1:
        st.write(df)

    with row1_2:
        number_features = len(X.columns)

        #feature with missing values
        drop_cols = []
        for col in X.columns:
            #put the feature in the drop trable if threshold not respected
            if((X[col].isna().sum()/len(X)*100 > missing_value_threshold_selected) & (col not in drop_cols)):
                drop_cols.append(col)


        #numerical columns
        num_cols_extracted = [col for col in X.select_dtypes(include='number').columns]
        num_cols = []
        num_cols_missing = []
        cat_cols = []
        cat_cols_missing = []
        for col in num_cols_extracted:
            if(len(X[col].unique()) < 25):
                cat_cols.append(col)
            else:
                num_cols.append(col)

        #categorical columns
        obj_cols = [col for col in X.select_dtypes(include=['object']).columns]
        text_cols = []
        text_cols_missing = []
        for col in obj_cols:
            if(len(X[col].unique()) < 25):
                cat_cols.append(col)
            else:
                text_cols.append(col)

        #remove dropped columns
        for element in drop_cols:
            if element in num_cols:
                num_cols.remove(element)
            if element in cat_cols:
                cat_cols.remove(element)
            if element in text_cols:
                text_cols.remove(element)

        #display info on dataset
        st.write('Original size of the dataset', X.shape)
        st.write('Dropping ', round(100*len(drop_cols)/number_features,2), '% of feature for missing values')
        st.write('Numerical columns : ', round(100*len(num_cols)/number_features,2), '%')
        st.write('Categorical columns : ', round(100*len(cat_cols)/number_features,2), '%')
        st.write('Text columns : ', round(100*len(text_cols)/number_features,2), '%')

        st.write('Total : ', round(100*(len(drop_cols)+len(num_cols)+len(cat_cols)+len(text_cols))/number_features,2), '%')

        #create new lists for columns with missing elements
        for col in X.columns:
            if (col in num_cols and X[col].isna().sum() > 0):
                num_cols.remove(col)
                num_cols_missing.append(col)
            if (col in cat_cols and X[col].isna().sum() > 0):
                cat_cols.remove(col)
                cat_cols_missing.append(col)
            # if (col in text_cols and X[col].isna().sum() > 0):
            #     text_cols.remove(col)
            #     text_cols_missing.append(col)

        #combine text columns in one new column because countVectorizer does not accept multiple columns
        X['text'] = X[text_cols].astype(str).agg(' '.join, axis=1)
        for cols in text_cols:
            drop_cols.append(cols)
        text_cols = 'text'


    #need to make two preprocessing pipeline too handle the case encoding without imputer...
    preprocessing = make_column_transformer(
        (get_pipeline_missing_cat(categorical_imputer_selected, encoder_selected) , cat_cols_missing),
        (get_pipeline_missing_num(numerical_imputer_selected, scaler_selected) , num_cols_missing),

        (get_encoding(encoder_selected), cat_cols),
        (get_encoding(text_encoder_selected), text_cols),
        (get_scaling(scaler_selected), num_cols)
    )


    dim = preprocessing.fit_transform(X).shape[1]
    if((encoder_selected == 'OneHotEncoder') | (dim > 2)):
        dim = dim - 1

    if (dim > 2):
        st.sidebar.title('Dimension Reduction')
        dimension_reduction_algorithm_selected = st.sidebar.selectbox('Algorithm', ['None', 'Kernel PCA'])

        hyperparameters_dim_reduc = {}                                      
        # if(dimension_reduction_algorithm_selected == 'PCA'):
        #     hyperparameters_dim_reduc['n_components'] = st.sidebar.slider('Number of components (default = nb of features - 1)', 2, dim, dim, 1)
        # if(dimension_reduction_algorithm_selected == 'LDA'):
        #     hyperparameters_dim_reduc['solver'] = st.sidebar.selectbox('Solver (default = svd)', ['svd', 'lsqr', 'eigen'])
        if(dimension_reduction_algorithm_selected == 'Kernel PCA'):
            hyperparameters_dim_reduc['n_components'] = st.sidebar.slider('Number of components (default = nb of features - 1)', 2, dim, dim, 1)
            hyperparameters_dim_reduc['kernel'] = st.sidebar.selectbox('Kernel (default = linear)', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
        # if(dimension_reduction_algorithm_selected == 'Truncated SVD'):
        #     hyperparameters_dim_reduc['n_components'] = st.sidebar.slider('Number of components (default = nb of features - 1)', 2, dim, dim, 1)
    else :
        st.sidebar.title('Dimension Reduction')
        dimension_reduction_algorithm_selected = st.sidebar.selectbox('Number of features too low', ['None'])
        hyperparameters_dim_reduc = {}         

    st.sidebar.title('Cross Validation')
    type = st.sidebar.selectbox('Type', ['KFold', 'StratifiedKFold'])
    nb_splits = st.sidebar.slider('Number of splits', min_value=3, max_value=20)
    folds = get_fold(type, nb_splits)

    st.sidebar.title('Model Selection')
    
    regressor_list = ['Linear Regression', 'SGD', 'K Nearest Neighbors', 'Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network']
    regressor_selected = st.sidebar.selectbox('', regressor_list)

    st.sidebar.header('Hyperparameters Selection')
    hyperparameters = {}

    if(regressor_selected == 'Linear Regression'):
        hyperparameters = {}

    if(regressor_selected == 'SGD'):
        hyperparameters['alpha'] = st.sidebar.slider('Alpha', 0.0, 0.0007, 0.0001, 0.0001)
        hyperparameters['loss'] = st.sidebar.selectbox('Loss', ['squared_loss', 'huber', 'epsilon_insensitive'])
        hyperparameters['penalty'] = st.sidebar.selectbox('Penalty', ['l2', 'l1', 'elasticnet'])
        hyperparameters['learning_rate'] = st.sidebar.selectbox('Learning rate', ['constant', 'optimal', 'invscaling'])


    if(regressor_selected == 'K Nearest Neighbors'):
        hyperparameters['n_neighbors'] = st.sidebar.slider('Number of neighbors (default value = 5)', 1, 21, 1, 5)
        hyperparameters['metric'] = st.sidebar.selectbox('Metric (default = minkowski)', ['minkowski', 'euclidean', 'manhattan'])
        hyperparameters['weights'] = st.sidebar.selectbox('Weights (default = uniform)', ['uniform', 'distance'])

        
    if(regressor_selected == 'Decision Tree'):
        hyperparameters['max_depth'] = st.sidebar.slider('Max Depth', 1, 20, 1, 2)
        hyperparameters['criterion'] = st.sidebar.selectbox('Criterion', ["mse", "mae"])
        hyperparameters['min_samples_leaf'] = st.sidebar.slider('Min sample leaf (default = 2)', 1, 10, 1, 1)

   
    if(regressor_selected == 'Random Forest'):
        hyperparameters['n_estimators'] = st.sidebar.slider('Number of estimators', 100, 600, 50, 100)       
        hyperparameters['criterion'] = st.sidebar.selectbox('Criterion (default = gini)', ['gini', 'entropy'])
        hyperparameters['min_samples_leaf'] = st.sidebar.slider('Min sample leaf', 1, 4, 1, 1)
        hyperparameters['max_depth'] = st.sidebar.slider('Max Depth', 1, 20, 1, 2)
        
    if(regressor_selected == 'XGBoost'):
        hyperparameters['n_estimators'] = st.sidebar.slider("n_estimators", 10, 400, step=50, value=40)
        hyperparameters['Learning Rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.3,value=0.04)
        hyperparameters['max_depth'] = st.sidebar.slider("max_depth", 1, 20, value=2)
        hyperparameters['Gamma'] = st.sidebar.slider("Gamma",0,10,value=8)
        hyperparameters['reg_lambda'] = st.sidebar.slider("reg_lambda",1.0,5.0,step=0.1)
        hyperparameters['reg_alpha'] = st.sidebar.slider("reg_alpha",0.0,5.0,step=0.1)
        hyperparameters['colsample_bytree'] = st.sidebar.slider("colsample_bytree",0.5,1.0,step=0.1)    
        
        
    if(regressor_selected == 'Neural Network'):
        hyperparameters['hidden_layer_sizes'] = st.sidebar.slider("hidden_layer_sizes", 1, 50, step=1, value=1)
        hyperparameters['activation'] = st.sidebar.selectbox("activation", ("identity", "logistic", "tanh", "relu"))
        hyperparameters['solver'] = st.sidebar.selectbox("solver", ("lbfgs", "sgd", "adam"))
        hyperparameters['alpha'] = st.sidebar.slider("alpha",0.00005,0.0005,step=0.00005)
    
    # In[ ]:


    
    preprocessing_pipeline = Pipeline([
        ('preprocessing' , preprocessing),
        ('dimension reduction', get_dim_reduc_algo(dimension_reduction_algorithm_selected, hyperparameters_dim_reduc))
    ])


    pipeline = Pipeline([
        ('preprocessing' , preprocessing),
        ('dimension reduction', get_dim_reduc_algo(dimension_reduction_algorithm_selected, hyperparameters_dim_reduc)),
        ('ml', get_reg_ml_algorithm(regressor_selected, hyperparameters))
    ])

    cv_score1 = cross_val_score(pipeline, X, Y, cv=folds, scoring='neg_mean_absolute_percentage_error')
    cv_score2 = cross_val_score(pipeline, X, Y, cv=folds, scoring='r2')
    cv_score3 = cross_val_score(pipeline, X, Y, cv=folds, scoring='neg_root_mean_squared_error')
    cv_score4 = cross_val_score(pipeline, X, Y, cv=folds, scoring='neg_mean_absolute_error')
    
    preprocessing_pipeline.fit(X)
    X_preprocessed = preprocessing_pipeline.transform(X)

    st.header('Preprocessed dataset')
    st.write(X_preprocessed)


    st.subheader('Results')
    st.write('MAPE : ', round(cv_score1.mean()*100*(-1),2), '%')
    st.write('RMSE : ', round(cv_score3.mean()*(-1),2))
    st.write('MAE : ', round(cv_score4.mean()*(-1),2))
    st.write('R-Squared : ', round(cv_score2.mean()*100,2), '%')   
    
    
    
    
page_names_to_funcs = {
    "Main Page": main_page,
    "Classification Modeling": page2,
    "Regression Modeling": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()