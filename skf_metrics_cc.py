import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
import math
import collections

#import scanpy as sc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

from mlens.ensemble import Subsemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def build_subsemble(proba):
    
    """Build a subsemble with random partitions"""
    sub = Subsemble(partitions=1, folds=10, verbose=10)
    sub.add([SVC(kernel='rbf', class_weight = 'balanced', probability=True),
             SVC(kernel='linear', class_weight = 'balanced', probability=True),
             SVC(kernel='poly', class_weight = 'balanced', probability=True),
             RandomForestClassifier(class_weight = 'balanced'),
             MLPClassifier(hidden_layer_sizes= (200, 200), max_iter=1000),
             XGBClassifier()
            ],
           proba=True) 
    sub.add_meta(SVC(kernel='rbf'))
    return sub


def score_model(model, X_train, y_train, name, out, n_splits = 10, upsample_multiple = 1, upsample = True, log = False, pca = False, cv="stratified", group=None):
    """
    Creates folds manually, and upsamples within each fold.
    Returns total performance metrics and class-specific performance metrics for each fold
    """
    
    label_set = list(set(y_train))
    label_value_counts_df = pd.Series(y_train).value_counts()
    y_mode = label_value_counts_df.idxmax()
    sampling_strategy_dict = {}
    for label in label_set:
        sampling_strategy_dict[label] = int(label_value_counts_df[y_mode] * upsample_multiple)
    sampling_strategy_dict[y_mode] = label_value_counts_df[y_mode]
    
    smoter = SMOTE(random_state = 0, n_jobs = -1)

    counter = 0
    results_df = pd.DataFrame(columns = ["Accuracy", "Precision", "Recall", "F1", "CK", "MCC", "Train_Time", "Test_Time"])

    if cv == "stratified":
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5).split(X_train, y_train)
    
    else:
        cv = LeaveOneGroupOut().split(X = X_train, y = y_train, groups = group)
        
    for train_fold_index, val_fold_index in cv:

        try:
            
            class_results_df = pd.DataFrame(index = list(set(y_train)), columns = ["Accuracy", "Precision", "Recall", "F1", "CK", "MCC"])
    
            # Get the training data
            X_train_fold, y_train_fold = X_train.iloc[train_fold_index], y_train[train_fold_index]
            # Get the validation data
            X_val_fold, y_val_fold = X_train.iloc[val_fold_index], y_train[val_fold_index]
            
            if log is True:
                # Log transform 
                X_val_fold = np.log10(X_val_fold + 1)
                X_train_fold = np.log10(X_train_fold + 1)
                
                print("Log Transform")
                
            if upsample is True:
                # Upsample only the data in the training section
                X_train_fold, y_train_fold = smoter.fit_resample(X_train_fold, y_train_fold)
                
                print("SMOTE Upsample")
            
            if pca is True:
                #PCA reduce 
                pca_test = PCA(n_components=0.999, svd_solver = 'full')
                X_val_fold = pca_test.fit_transform(X_val_fold)
    
                pca_train = PCA(n_components=pca_test.n_components_, svd_solver = 'full')
                X_train_fold = pca_train.fit_transform(X_train_fold)
                
                print("PCA Reduce")
     
            # Fit the model on the upsampled training data
            train_start_time = time.time()
            model_obj = model.fit(X_train_fold, y_train_fold)
            results_df.loc[counter, 'Train_Time'] = time.time() - train_start_time
            
            print("Fit")
    
            # Score the model on the (non-upsampled) validation data
            test_start_time = time.time()
            model_predictions = model_obj.predict(X_val_fold)
            
            results_df.loc[counter, 'Test_Time'] = time.time() - test_start_time
            results_df.loc[counter, 'Accuracy'] = accuracy_score(y_val_fold, model_predictions)
            results_df.loc[counter, 'F1'] = f1_score(y_val_fold, model_predictions,average='weighted')
            results_df.loc[counter, 'Precision'] = precision_score(y_val_fold, model_predictions,average='weighted')
            results_df.loc[counter, 'Recall'] = recall_score(y_val_fold, model_predictions,average='weighted')
            results_df.loc[counter, 'CK'] = cohen_kappa_score(y_val_fold, model_predictions)
            results_df.loc[counter, 'MCC'] = matthews_corrcoef(y_val_fold, model_predictions)
        
            for label in label_set:
                class_results_df.loc[label, 'Accuracy'] = accuracy_score(np.repeat(label, len(model_predictions[[list(np.where(y_val_fold == label)[0])]])), 
                                                                   model_predictions[[list(np.where(y_val_fold == label)[0])]])
                class_results_df.loc[label, 'F1'] = f1_score(np.repeat(label, len(model_predictions[[list(np.where(y_val_fold == label)[0])]])), 
                                                                   model_predictions[[list(np.where(y_val_fold == label)[0])]],
                                                                   average='weighted')
                class_results_df.loc[label, 'Precision'] = precision_score(np.repeat(label, len(model_predictions[[list(np.where(y_val_fold == label)[0])]])), 
                                                                    model_predictions[[list(np.where(y_val_fold == label)[0])]],
                                                                   average='weighted')
                class_results_df.loc[label, 'Recall'] = recall_score(np.repeat(label, len(model_predictions[[list(np.where(y_val_fold == label)[0])]])), 
                                                                    model_predictions[[list(np.where(y_val_fold == label)[0])]],
                                                                   average='weighted')
                class_results_df.loc[label, 'CK'] = cohen_kappa_score(np.repeat(label, len(model_predictions[[list(np.where(y_val_fold == label)[0])]])), 
                                                                    model_predictions[[list(np.where(y_val_fold == label)[0])]])
                class_results_df.loc[label, 'MCC'] = matthews_corrcoef(np.repeat(label, len(model_predictions[[list(np.where(y_val_fold == label)[0])]])), 
                                                                    model_predictions[[list(np.where(y_val_fold == label)[0])]])
    
                print("Test Class")
                
            class_results_df.to_csv(out + "/" + name  + "_fold_" + str(counter) + "_label_metrics.tsv", sep="\t")
            counter += 1
            
        except:
            pass

    results_df.to_csv(out + "/" + name  + "_total_metrics.tsv", sep='\t')
    
def format_data(count, label, name, out):
        # Read data as dataframe
        X = pd.read_csv(count, index_col=0)
        X.fillna(0, inplace=True)
        y_table = pd.read_csv(label)
        group = y_table['patient']
        
        #Map labels to numbers
        label_df = pd.DataFrame(columns=['Cell_Type'])
        label_df['Cell_Type'] = y_table['truth'].astype('category')
        label_df['Cell_Type_Cat'] = label_df['Cell_Type'].cat.codes
        
        label_df.to_csv(out + "/" + name + "_label_df.tsv", sep="\t")
        
        map_label_df = label_df.drop_duplicates().sort_values("Cell_Type_Cat").reset_index(drop=True)
        map_label_df.to_csv(out + "/" + name + "_map_label_df.tsv", sep="\t")
        y = label_df['Cell_Type_Cat'].astype('category').values
        
        return X, y, group
    
def main(): 
    
    # Initiate the parse
    #parser = argparse.ArgumentParser(description="Metrics parameters")
    
    # Add long and short argument
    #parser.add_argument("--out", "-o", help="Output directory")
    #parser.add_argument("--count", "-c", help="Count file")
    #parser.add_argument("--label", "-l", help="Label file")
    #parser.add_argument("--name", "-n", help="Dataset name")
    
    # Read arguments from the command line
    #args = parser.parse_args()
    #out = str(args.out)
    #count = str(args.count)
    #label = str(args.label)
    #name = str(args.name)
    
    out = "/home/dc0420/projects/def-pshoosht/dc0420/vangalen_aml"
    count = "/home/dc0420/projects/def-pshoosht/dc0420/datasets/vangalen_aml_counts.csv"
    label = "/home/dc0420/projects/def-pshoosht/dc0420/datasets/vangalen_aml_labels.csv"
    name = "vangalen_aml"
    # Create output directory if does not exist
    try:
        os.mkdir(out)
    except:
        print ("Creation of the directory failed. Check if it already exists!")
    
    
    X, y, group = format_data(count, label, name, out)
    
    # Instantiate Subsemble
    subsemble_model = build_subsemble(proba=True)
    
    #score_model(X_train = X, y_train = y, model = subsemble_model, name = "Subsemble_10", out = out, n_splits=10, upsample_multiple = 1, upsample = False, log = False, pca = False)
    score_model(X_train = X, y_train = y, model = subsemble_model, name = "Subsemble_LOOCV", out = out, n_splits=10, upsample_multiple = 1, upsample = False, log = False, pca = False, cv='patient', group=group)
    score_model(X_train = X, y_train = y, model = subsemble_model, name = "Subsemble_5", out = out, n_splits=5, upsample_multiple = 1, upsample = False, log = False, pca = False)
    score_model(X_train = X, y_train = y, model = subsemble_model, name = "Subsemble_20", out = out, n_splits=20, upsample_multiple = 1, upsample = False, log = False, pca = False)
   
if __name__ ==  '__main__':
    main() 

        
    
    