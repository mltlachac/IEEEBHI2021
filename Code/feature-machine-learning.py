# -*- coding: utf-8 -*-
"""
@author: Veronica Melican
"""

from collections import Counter
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--modalities', type=str, nargs = '+', required = True,
        help = "List of modalities to use, either 'text' or 'call'")
    arg_parser.add_argument('--directions', type=str, nargs = '+', required = True,
        help = "List of directions to use, either 'All', 'In', or 'Out'")
    arg_parser.add_argument('--days', type=int, nargs = '+', required = True,
        help = "List of lengths of time series data in days")
    arg_parser.add_argument('--intervals', type=int, nargs = '+', required = True,
        help = "List of lengths of time series intervals in hours")
    arg_parser.add_argument('--variables', type=str, nargs = "+", required = True,
        help = "List of variables")
    arg_parser.add_argument('--methods', type=str, nargs = "+", required = True, choices = ['SVC', 'RFC', 'KNN', 'LR', 'XG'],
        help = "List of machine learning methods to use")
    arg_parser.add_argument('--samplings', type=str, nargs = "+", required = True, choices = ['up', 'down'],
        help = "List of data balancing techniques to use, either 'up' or 'down'")
    arg_parser.add_argument('--numsplits', type=int, required = True,
        help = "Number of train test splits to run for each set of parameters")
    arg_parser.add_argument('--maxpc', type=int, required = True,
        help = "Maximum number of principal components to check")
    arg_parser.add_argument('--inputPath', type=str, required = True,
        help = "Path to the folder with input features")
    arg_parser.add_argument('--outputFile', type=str, required = True,
        help = "Name to use for results files")
    return arg_parser
    
def main(args):
    modalities = args.modalities        
    directions = args.directions
    days = args.days
    intervals = args.intervals
    variables = args.variables
    methods = args.methods
    samplings = args.samplings
    numsplits = args.numsplits
    maxpc = args.maxpc
    inputPath = args.inputPath
    outputFile = args.outputFile
    
    results = []
    
    for modality in modalities:
        for direction in directions:
            for day in days:
                for variable in variables:
                    for method in methods:
                        for sampling in samplings:
                            #print(modality, direction, day, variable, method, sampling)
                            results += run(modality, direction, day, intervals, variable, method, sampling, numsplits, maxpc, inputPath)
                           
    #combine results into dataframes         
    combined = pd.concat(results)
    combined.to_csv(outputFile + "Results.csv")
    
    #create summary scores for each parameter set
    summary = pd.DataFrame();
    grouped = combined.groupby(['modality','direction','day','interval','variable', 'pc', 'method', 'sampling'])
    
    summary['f1_mean'] = grouped['f1'].mean()
    summary['f1_stdev'] = grouped['f1'].std()
    summary['accuracy_mean'] = grouped['accuracy'].mean()
    summary['accuracy_stdev'] = grouped['accuracy'].std()
    summary['precision_mean'] = grouped['precision'].mean()
    summary['precision_stdev'] = grouped['precision'].std()
    summary['sensitivity_mean'] = grouped['sensitivity'].mean()
    summary['sensitivity_stdev'] = grouped['sensitivity'].std()
    summary['specificity_mean'] = grouped['specificity'].mean()
    summary['specificity_stdev'] = grouped['specificity'].std()
    summary['auc_mean'] = grouped['auc'].mean() 
    summary['auc_stdev'] = grouped['auc'].std()
    
    summary_sorted = summary.sort_values(by='f1_mean', ascending=False)
    summary_sorted.to_csv(outputFile + "Summary.csv")

def run(modality, direction, day, intervals, variable, method, sampling, numsplits, maxpc, inputPath):
    
    #read file to use to create indices for kfold split
    df = pd.read_csv(inputPath + '/' + modality + direction + str(day) + '_' + str(intervals[0]) + '_' + variable + '.csv')
    
    splits = StratifiedShuffleSplit(n_splits=numsplits, test_size=.33, random_state=100)
    splits.get_n_splits(df)
    
    results = []    
    pcs = list(range(1, maxpc + 1));
    
    #iterate through intervals and k_values
    for interval in intervals:
        for pc in pcs:
            print(modality, direction, day, interval, variable, method, sampling, pc)
            #read files
            df = pd.read_csv(inputPath + '/' + modality + direction  + str(day) + '_' + str(interval) + '_' + variable +  '.csv')
        
            #split into features and target
            y = df['depressed']
            X = df.drop(['depressed'], axis=1)
            
            #drop columns that contains infinity
            X.replace(float('inf'), np.nan, inplace=True)
            X.replace(np.inf, np.nan, inplace=True)
            X.dropna(axis=1, how="any", inplace=True)   
            
            f1_list = []
            accuracy_list = []
            precision_list = []
            sensitivity_list= []
            specificity_list = []
            auc_list = []
            ytrue_list = []
            yhat_list = []
            splitnum_list = []
            modality_list = []
            direction_list = []
            day_list= []
            interval_list = []
            variable_list = []
            method_list = []
            sampling_list = []
            pc_list = []
                
            #run for each split
            count = 0
            for train_index, test_index in splits.split(X, y):
                count += 1
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                #scale   
                cols = X.columns
                indextrain = X_train.index
                indextest = X_test.index
                
                scaler = MinMaxScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                
                X_train = pd.DataFrame(X_train, columns=cols, index=indextrain)
                X_test = pd.DataFrame(X_test, columns=cols, index=indextest)
                
                #PCA            
                pca = PCA(n_components=pc, random_state=100);
                pca = pca.fit(X_train);
                
                X_train = pd.DataFrame(pca.transform(X_train), index=indextrain)
                X_test = pd.DataFrame(pca.transform(X_test), index=indextest)
                
                #downsample/upsample
                if sampling == "up":
                    X_train, y_train = upsample(X_train, y_train)
                if sampling == "down":
                    X_train, y_train = downsample(X_train, y_train)
                
                #train model
                if method == "SVC":
                    clf = svm.SVC()
                elif method == "RFC":
                    clf = RandomForestClassifier()
                elif method == "KNN":
                    clf = KNeighborsClassifier(n_neighbors=3)
                elif method == "LR":
                    clf = LogisticRegression()
                clf.fit(X_train, y_train)
                
                #test model
                y_pred = clf.predict(X_test)
                
                #get scoring metrics
                f1, accuracy, precision, sensitivity, specificity, auc = get_scores(y_test, y_pred)
                f1_list.append(f1)
                accuracy_list.append(accuracy)
                precision_list.append(precision)
                sensitivity_list.append(sensitivity)
                specificity_list.append(specificity)
                auc_list.append(auc)
                ytrue_list.append(y_test.values)
                yhat_list.append(y_pred)
                
                #add parameter information
                splitnum_list.append(count)
                modality_list.append(modality)
                direction_list.append(direction)
                day_list.append(day)
                interval_list.append(interval)
                variable_list.append(variable)
                pc_list.append(pc)
                method_list.append(method)
                sampling_list.append(sampling)
        
            #create dataframe of results
            result = pd.DataFrame()
            result['modality'] = modality_list
            result['direction'] = direction_list
            result['day'] = day_list
            result['interval'] = interval_list
            result['variable'] = variable_list
            result['pc'] = pc_list
            result['method'] = method_list
            result['sampling'] = sampling_list
            result['split_number'] = splitnum_list
            result['f1'] = f1_list
            result['accuracy'] = accuracy_list
            result['precision'] = precision_list
            result['sensitivity'] = sensitivity_list
            result['specificity'] = specificity_list
            result['auc'] = auc_list
            result['ytrue'] = ytrue_list
            result['yhat'] = yhat_list
            
            results.append(result.copy())
            
    return results

def downsample(X_train, y_train):
        
    train = X_train.copy()
    train['y'] = y_train
    
    rus = RandomUnderSampler()
    X_train, y_train = rus.fit_sample(
        train.drop('y', axis=1), 
        train['y'])
    
    return X_train, y_train  

def upsample(X_train, y_train):
    
    train = X_train.copy()
    train['y'] = y_train
    
    ros = RandomOverSampler()
    X_train, y_train = ros.fit_sample(
        train.drop('y', axis=1), 
        train['y'])
    
    return X_train, y_train

def get_scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = 2*tp / (2*tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    auc = roc_auc_score(y_true, y_pred)
    
    return f1, accuracy, precision, sensitivity, specificity, auc

if __name__=="__main__":
    args = parser().parse_args()
    main(args)
