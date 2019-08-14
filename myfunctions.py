
# coding: utf-8

# In[ ]:


"""
Created on Tue Aug 06 11:08:51 2019

@author: p901cyo
"""
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.utils import resample

# ===================================
# 
# ==================================

def sumvar(df, varlist, newvar):
    df_1 = (df.fillna(0)
            .withColumn(newvar, sum(F.col(col) for col in varlist)).select('partygenid','disbursement_month',newvar))
    return df_1


# ===================================
# 
# ==================================


def bin_iv_var(df_pd, sel_var, grp_var, grp_name, var, default_n, nondefault_n, cell):
    # calculate IV function
    df_pd[grp_name] = pd.qcut(df_pd[var].astype('float'),cell,labels=False)
    df_pd.fillna(-1,inplace=True)
    summary = df_pd[sel_var].groupby(grp_var, as_index=False).agg(['mean','count'])
    df_pd_cal=(df_pd
               .groupby(grp_var, as_index=False)
               .agg({ var : ['max','min'],'default' : ['sum','count']}))
    df_pd_cal['Event_pct']=df_pd_cal['default']['sum']/default_n
    df_pd_cal['Nonevent_pct']=(df_pd_cal['default']['count']-df_pd_cal['default']['sum'])/nondefault_n
    df_pd_cal['WOE']=np.log(df_pd_cal['Event_pct']/df_pd_cal['Nonevent_pct'])
    df_pd_cal['Event_Nonevent_pct']=df_pd_cal['Event_pct']-df_pd_cal['Nonevent_pct']
    df_pd_cal['IV']=df_pd_cal['WOE']*df_pd_cal['Event_Nonevent_pct']
    df_pd_cal['var_name']=var
    #df_pd_cal.head(5)
    return summary, df_pd_cal;


# ===================================
# 
# ==================================

def bin_plot(data, var):
    # plotting function
    plt.clf
    fig = plt.figure()
    plt.xticks(data.index, figure=fig)
    plt.ylabel('default_rate', figure=fig)
    plt.title('bin of %s vs default rate' %var, figure=fig)
    plt.bar(data.index.astype('float'),data['default']['mean'], figure=fig)
    #plt.show()
    plt.close()
    return fig


# ===================================
# 
# ==================================


def optimal_binning_boundary(df, var):
    # get bin boundary from decision tree
    from sklearn.tree import DecisionTreeClassifier

    boundary = []  # return the boundaries
    
    x = df.select(var).toPandas()  # send to pandas
    y = df.select('default').toPandas()
    
    clf = DecisionTreeClassifier(criterion='entropy',   
                                 max_leaf_nodes=6,       
                                 min_samples_leaf=0.05)

    clf.fit(x, y)  
    
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  
            boundary.append(threshold[i])

    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1 is to make sure after groupby the maximum sample value will be included
    boundary.append(min_x[0])
    boundary.append(max_x[0])
    boundary.sort()
    

    return boundary


# ===================================
# 
# ==================================


def bin_iv_var2(df, var):
    # calculate information based on decision tree binning boundaries
    x = df.select(var).toPandas()  # send to pandas
    y = df.select('default').toPandas()
    boundary = optimal_binning_boundary(df=df, var=var)       
    df_pd = pd.concat([x, y], axis=1)                        
    df_pd.columns = ['x', 'y']                               
    df_pd['bins'] = pd.cut(x=x[var], bins=boundary, right=False)  
    
    grouped = df_pd.groupby('bins')['y']                     
    result_df = grouped.agg([('good',  lambda y: (y == 0).sum()), 
                             ('bad',   lambda y: (y == 1).sum()),
                             ('total', 'count')])

    result_df['good_pct'] = result_df['good'] / result_df['good'].sum()       
    result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()          
    result_df['total_pct'] = result_df['total'] / result_df['total'].sum()    

    result_df['bad_rate'] = result_df['bad'] / result_df['total']             
    
    result_df['woe'] = np.log(result_df['good_pct'] / result_df['bad_pct'])              
    result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe'] 
    
    return result_df


# ===================================
# 
# ==================================
def getfeature(file_name):
    feature_used_pd = pd.read_csv(file_name, sep=",", header=0) 
    features = list(feature_used_pd['var'][feature_used_pd['iv']>=0])
    return features

# ===================================
# 
# ==================================
def sample_method (method, n_ratio, sample_vector, stay_vector):
    # up and down sample
    if method=='upsampled':
        # Upsample minority class
        df_minority_upsampled = resample(sample_vector, 
                                         replace=True,     # sample with replacement
                                         n_samples= stay_vector.shape[0]/n_ratio,    # to match majority class
                                         random_state=123) # reproducible results

        # Combine majority class with upsampled minority class
        method = pd.concat([stay_vector, df_minority_upsampled])
    elif method=='downsampled':
        # Downsample majority class
        df_majority_downsampled = resample(sample_vector, 
                                 replace=False,    # sample without replacement
                                 n_samples=stay_vector.shape[0]*n_ratio,     # to match minority class
                                 random_state=123) # reproducible results
 
        # Combine minority class with downsampled majority class
        method = pd.concat([df_majority_downsampled, stay_vector])
    return method


