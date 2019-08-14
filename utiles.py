# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:07:27 2019
list of function to be used for preprocessing and modeling
@author: p901aze
"""
from pyspark.sql.functions import *
import pandas as pd
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

# ===================================
# 
# ==================================
def print_basic_stat(df):
    """
    Input: Pyspark dataframe
    """
    print('Number of records:')
    print(df.count())
    print('Number of features:')
    print(np.shape(df.columns)[0])
    print(df.groupby('default').count().show())

# ===================================
# 
# ==================================
def overview(df):
    data_overview = pd.DataFrame(columns = ['data type','percentage of missing'], index = df.columns.values)

    for col in range(len(df.columns)):
        index = str(df.columns.get_values()[col])
        missing = np.float(df.iloc[:,col].isnull().sum())/len(df.iloc[:,col])*100
        types = df.iloc[:,col].dtype
        data_overview.set_value(index, 'percentage of missing', missing)
        data_overview.set_value(index, 'data type', types)

    return(data_overview)

# ===================================
# 
# ==================================
def test_train_split(df, train_end_date, test_end_date, random):
    if random == 0: 
        df_train = df[df.disbursement_month < train_end_date].drop('disbursement_month', axis = 1) 
        df_test = df[(df.disbursement_month >= train_end_date) \
                     & (df.disbursement_month < test_end_date)].drop('disbursement_month', axis = 1)
        X_train = df_train.drop(['default'], axis = 1)
        y_train = df_train['default']
        X_test = df_test.drop(['default'], axis = 1)
        y_test = df_test['default']
        df_train = pd.concat([X_train, y_train],axis=1)
        df_test = pd.concat([X_test, y_test],axis=1)
    
    elif random == 1:
        X = df.drop(['default','disbursement_month'], axis= 1)
        y = df['default']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        df_train = pd.concat([X_train, y_train],axis=1)
        df_test = pd.concat([X_test, y_test],axis=1)
        
    return df_train, df_test

# ===================================
# 
# ==================================
def CreateHeatMap(df):
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

# ===================================
# 
# ==================================
def plot_feature_importances(model,n,X_train):
    importances = model.named_steps['clf'].best_estimator_.feature_importances_

    indices = np.argsort(importances)[::-1]
    indices = indices[0:n]
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(0,10), importances[indices],
           color="r", align="center")
    plt.yticks(range(0,10), X_train.columns[indices])
    plt.ylim([-1, 10])
    plt.show()    
    
# ===================================
# 
# ==================================
def plot_classes_in_pc_space(pc1, pc2, minimum_n, X_train_s, y_train ):
    pca = PCA(n_components = minimum_n)
    X_pca = pca.fit_transform(X_train_s)[:,[pc1,pc2]]
    zero_class = np.where(y_train == 0)
    one_class = np.where(y_train == 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X_pca[zero_class, 0], X_pca[zero_class, 1], s=16, edgecolors='b',
                facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X_pca[one_class, 0], X_pca[one_class, 1], s=8, edgecolors='orange',
                facecolors='none', linewidths=2, label='Class 2')
    plt.xlabel('Principal component number '+str(pc1))
    plt.ylabel('Principal component number '+str(pc2))
    plt.legend(loc="upper left")
    plt.show()
    
# ===================================
# 
# ==================================   
def pca_feature_importance_v(minimum_n, X_train_s, df_train ):
    pca = PCA(n_components = minimum_n)
    pca.fit(X_train_s)
    plt.figure(figsize=(30,30))
    plt.matshow(pca.components_.transpose(),cmap='viridis', fignum=1)
    plt.xticks(range(minimum_n), range(minimum_n), fontsize=10)
    plt.colorbar()
    plt.yticks(range(len(df_train.drop('default', axis = 1).columns)),df_train.drop('default', axis = 1).columns)#,rotation=90,ha='left')
    plt.tight_layout()
    plt.show()#


# ===================================
# 
# ==================================   
def pca_feature_importance_h(minimum_n, X_train_s, df_train ):
    pca = PCA(n_components = minimum_n)
    pca.fit(X_train_s)
    plt.figure(figsize=(20,5))
    plt.matshow(pca.components_,cmap='viridis', fignum=1)
    plt.yticks(range(minimum_n), range(minimum_n), fontsize=10)
    plt.colorbar()
    plt.xticks(range(len(df_train.drop('default', axis = 1).columns)),df_train.drop('default', axis = 1).columns,rotation=60,ha='left')
    plt.tight_layout()
    plt.show()#    
    
    


# ===================================
# 
# ==================================
def sample_method (method, n_ratio, sample_vector, stay_vector):
    if method=='random_oversampled':
        # Upsample minority class
        df_minority_upsampled = resample(sample_vector, 
                                         replace=True,     # sample with replacement
                                         n_samples= stay_vector.shape[0]//n_ratio,    # to match majority class
                                         random_state=123) # reproducible results

        # Combine majority class with upsampled minority class
        df_train = pd.concat([stay_vector, df_minority_upsampled])
        X_train_res = df_train.drop(['default'], axis=1)
        y_train_res = df_train['default']

    elif method=='random_downsampled':
        # Downsample majority class
        df_majority_downsampled = resample(sample_vector, 
                                 replace=False,    # sample without replacement
                                 n_samples=stay_vector.shape[0]*n_ratio,     # to match minority class
                                 random_state=123) # reproducible results
 
        # Combine minority class with downsampled majority class
        df_train = pd.concat([df_majority_downsampled, stay_vector])
        X_train_res = df_train.drop(['default'], axis=1)
        y_train_res = df_train['default']
        

    elif method=='SMOTE_oversampled':
        from imblearn.over_sampling import SMOTE
        method_over = SMOTE(kind='regular')
        df_train = pd.concat([sample_vector, stay_vector])
        X_train_res, y_train_res = method_over.fit_sample(df_train.drop('default', axis = 1), df_train.default)
        
    elif method == 'EditedNearestNeighbours':
        # This is not working!!
        method_down = EditedNearestNeighbours(sampling_strategy = 'majority')
        df_train = pd.concat([sample_vector, stay_vector])
        X_train_res, y_train_res = method_down.fit_sample(df_train.drop('default', axis = 1), df_train.default)
        
    elif method == 'combined':
        method_resample = SMOTEENN(sampling_strategy = 1)
        df_train = pd.concat([sample_vector, stay_vector])
        X_train_res, y_train_res = method_resample.fit_sample(df_train.drop('default', axis = 1), df_train.default)
        
    #print('Total number of samples after resampling: '+str(np.shape(y_train_res)[0]))
    #print('default percentage in the train dataset after re-sampling is: ' + str(sum(y_train_res)//len(y_train_res)*100))  
    return X_train_res, y_train_res



# ===================================
# CHEN
# ==================================
def sumvar(df, varlist, newvar):
    from functools import reduce
    from operator import add
    df_1 = df.na.fill(0).withColumn(newvar ,reduce(add, [col(x) for x in varlist]))
    df_2 = df_1.select('partygenid','disbursement_month',newvar)
    return df_2

# ===================================
# 
# ==================================
def get_list_type(df,data_type):
    list_name = []
    for x, t in df.dtypes:
        if data_type in t:
            list_name.append(x)
    return list_name

# ===================================
# 
# ==================================
def get_list_of_unique_type(df):
    list_type_unique = []
    for x, t in df.dtypes:
        if t not in list_type_unique:
            list_type_unique.append(t)
        
    return(list_type_unique)




# ===================================
# Bing
# ==================================
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# ===================================
# Bing
# ==================================
def get_top_abs_correlations(df, threshold):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    au_corr = au_corr[au_corr > threshold]
    
    au_corr_df = au_corr.to_frame().reset_index()
    au_corr_df = au_corr_df.rename(columns ={'level_0': 'feature_1',
                                             'level_1': 'feature_2',
                                              0: 'correlation'})
    print 'The number of highly correlated feature pairs is: ', au_corr_df['correlation'].count()
    return au_corr_df
