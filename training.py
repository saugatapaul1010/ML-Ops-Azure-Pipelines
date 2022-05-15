import os
import sys
import azureml as aml
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.core.run import Run
import argparse
import json
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import math
import seaborn as sn
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
class AzureClassification():

    def __init__(self, args):
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
            2. Create directories
        '''
        self.args = args
        self.run = Run.get_context()
        self.workspace = self.run.experiment.workspace
        os.makedirs('./model_metas', exist_ok=True)

    def create_classification_pipeline(self):
        '''
        Data training and Validation
        '''
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        input_ds = self.get_files_from_datastore(self.args.container_name,self.args.input_csv)
        final_df = input_ds.to_pandas_dataframe()
        print("Input DF Info",final_df.info())
        print("Input DF Head",final_df.head())
        print(final_df.columns)
        print(final_df.head())

        X = final_df[self.args.training_columns.split(",")]
        y = final_df[[self.args.target_column]]
    
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1-self.args.train_size,random_state=1984)
        # print("Selecting Best Model")
        models = [
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0,n_jobs=-1),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(random_state=0,n_jobs=-1),
            ExtraTreesClassifier(n_estimators=100, random_state=0,n_jobs=-1),
            DecisionTreeClassifier()
        ]
    
        # 5 Cross-validation
        CV = 5
        cv_df = pd.DataFrame(index=range(CV * len(models)))
    
        entries = []
        model_names=[]
        for model in models:
            model_name = model.__class__.__name__
            model_names.append(model_name)
            accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
    
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
        mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
        std_accuracy = cv_df.groupby('model_name').accuracy.std()
    
        acc = pd.concat([mean_accuracy, std_accuracy], axis= 1,
                ignore_index=True)
        acc.columns = ['Mean Accuracy', 'Standard deviation']
        print("Best Model Selected {}".format(acc['Mean Accuracy'].idxmax()))
        model = models[model_names.index(acc['Mean Accuracy'].idxmax())]
        # model = DecisionTreeClassifier()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print("Model Score : ", model.score(X_test,y_test))
    
        joblib.dump(model, self.args.model_path)
    
        self.validate(y_test, y_pred, X_test)
    
        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)
    
        print("Run Files : ", self.run.get_file_names())
        self.run.complete()

    def create_confusion_matrix(self, y_true, y_pred, name):
        '''
        Create confusion matrix
        '''
        try:
            confm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
            print("Shape : ", confm.shape)
    
            df_cm = pd.DataFrame(confm, columns=np.unique(y_true), index=np.unique(y_true))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            df_cm.to_csv(name+".csv", index=False)
            self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")
    
            plt.figure(figsize = (120,120))
            sn.set(font_scale=1.4)
            c_plot = sn.heatmap(df_cm, fmt="d", linewidths=.2, linecolor='black',cmap="Oranges", annot=True,annot_kws={"size": 16})
            plt.savefig("./outputs/"+name+".png")
            self.run.log_image(name=name, plot=plt)
        except Exception as e:
            #traceback.print_exc()
            logging.error("Create consufion matrix Exception")

    def create_outputs(self, y_true, y_pred, X_test, name):
        '''
        Create prediction results as a CSV
        '''
        pred_output = {"Actual "+self.args.target_column : y_true[self.args.target_column].values, "Predicted "+self.args.target_column: y_pred[self.args.target_column].values}
        pred_df = pd.DataFrame(pred_output)
        pred_df = pred_df.reset_index()
        X_test = X_test.reset_index()
        final_df = pd.concat([X_test, pred_df], axis=1)
        final_df.to_csv(name+".csv", index=False)
        self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

    def get_files_from_datastore(self, container_name, file_name):
        '''
        Get the input CSV file from workspace's default data store
        Args :
            container_name : name of the container to look for input CSV
            file_name : input CSV file name inside the container
        Returns :
            data_ds : Azure ML Dataset object
        '''
        datastore_paths = [(self.datastore, os.path.join(container_name,file_name))]
        data_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
        dataset_name = self.args.dataset_name
        if dataset_name not in self.workspace.datasets:
            data_ds = data_ds.register(workspace=self.workspace,
                        name=dataset_name,
                        description=self.args.dataset_desc,
                        tags={'format': 'CSV'},
                        create_new_version=True)
        else:
            print('Dataset {} already in workspace '.format(dataset_name))
        return data_ds

    def validate(self, y_true, y_pred, X_test):
        self.run.log(name="Precision", value=round(precision_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Recall", value=round(recall_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Accuracy", value=round(accuracy_score(y_true, y_pred), 2))
    
        self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")
    
        y_pred_df = pd.DataFrame(y_pred, columns = [self.args.target_column])
        self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        self.run.tag(self.args.tag_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QA Code Indexing pipeline')
    parser.add_argument('--container_name', type=str, help='Path to default datastore container')
    parser.add_argument('--input_csv', type=str, help='Input CSV file')
    parser.add_argument('--dataset_name', type=str, help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', type=str, help='Dataset description')
    parser.add_argument('--model_path', type=str, help='Path to store the model')
    parser.add_argument('--artifact_loc', type=str,help='DevOps artifact location to store the model', default='')

    parser.add_argument('--training_columns', type=str, help='model training columns comma separated')
    parser.add_argument('--target_column', type=str, help='target_column of model prediction')
    parser.add_argument('--train_size', type=float, help='train data size percentage. Valid values can be 0.01 to 0.99')
    parser.add_argument('--tag_name', type=str, help='Model Tag name')
    parser.add_argument('--balancing_technique_technique', type=str, help='Available options RUS,ROS and SMOTE')
    parser.add_argument('--preprocess_remove_whitespace', type=bool, help='Remove white space from text. True or False. default True.',)
    parser.set_defaults(preprocess_remove_whitespace=False)
    parser.add_argument('--preprocess_remove_hyperlinks', type=bool, help='Remove hyperlinks from text. True or False. default True.')
    parser.set_defaults(preprocess_remove_hyperlinks=False)
    parser.add_argument('--preprocess_remove_htmltags', type=bool, help='Remove htmltags from text. True or False. default True.')
    parser.set_defaults(preprocess_remove_htmltags=False)
    parser.add_argument('--preprocess_expand_words', type=bool, help='Remove expand contracted words from text. ex: won\'t ==> will not. True or False. default True.')
    parser.set_defaults(preprocess_expand_words=False)
    parser.add_argument('--preprocess_remove_numericdata', type=bool, help='Remove numeric data from text. True or False. default True.')
    parser.set_defaults(preprocess_remove_numericdata=False)
    parser.add_argument('--preprocess_remove_anyspecialchars', type=bool, help='Remove any special characters apart from A-Z from text. True or False. default True.')
    parser.set_defaults(preprocess_remove_anyspecialchars=False)
    parser.add_argument('--preprocess_remove_stopwords', type=bool, help='Remove stopwords from text. True or False. default True.')
    parser.set_defaults(preprocess_remove_stopwords=False)
    parser.add_argument('--preprocess_strip', type=bool, help='Remove extra spaces from text. True or False. default True.')
    parser.set_defaults(preprocess_strip=False)
    args = parser.parse_args()
    classifier = AzureClassification(args)
    classifier.create_classification_pipeline()
