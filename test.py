from pathlib import Path 
import numpy as np 
import os
from arff2pandas import a2p
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import cohen_kappa_score

import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.core.classes import Random
from weka.filters import Filter

def file_find(base_dir, file_ext="arff"):
    paths = []
    for base, directories, files in os.walk(base_dir):
        for file_name in files:
            if("."+file_ext in file_name):
                paths.append(base+"/"+file_name)
    return paths

def load_arff(infile_path):
    f = open(infile_path)
    #print(infile_path)
    dataframe = a2p.load(f)
    dataframe = dataframe.rename(index=str, columns={dataframe.columns[-1]: 'target'})
    return dataframe

def _dtype_is_numeric(dtype):
        return "int" in str(dtype) or "float" in str(dtype)

def _preprocess_data(dataframe):
        series_array = []
        for feature in dataframe.columns:
            feature_series = dataframe[feature]
            col = feature_series.as_matrix()
            dropped_nan_series = feature_series.dropna(axis=0,how='any')
            num_nan = col.shape[0] - dropped_nan_series.shape[0]
            if num_nan != col.shape[0]:
	            col[feature_series.isnull()] = np.random.choice(dropped_nan_series, num_nan)
	            if not _dtype_is_numeric(feature_series.dtype):
	                feature_series = pd.get_dummies(feature_series)
	            series_array.append(feature_series)
        preprocessed_dataframe = pd.concat(series_array, axis=1, copy=False)
        return preprocessed_dataframe

files = file_find("./datasets")
files.sort()

weka_auc = {}
weka_kappa = {}
weka_err = {}
sklearn_auc = {}
sklearn_kappa = {}
sklearn_err = {}
weka_results = {"AUC" : weka_auc, "KAPPA" : weka_kappa, "ERR" : weka_err}
sklearn_results = {"AUC" : sklearn_auc, "KAPPA" : sklearn_kappa, "ERR" : sklearn_err}

jvm.start(max_heap_size="8192m")

rt1 = Classifier(classname="weka.classifiers.trees.RandomTree", options=["-depth", "1"])
rt2 = Classifier(classname="weka.classifiers.trees.RandomTree", options=["-depth", "2"])
rt3 = Classifier(classname="weka.classifiers.trees.RandomTree", options=["-depth", "3"])
rep1 = Classifier(classname="weka.classifiers.trees.REPTree", options=["-L", "1"])
rep2 = Classifier(classname="weka.classifiers.trees.REPTree", options=["-L", "2"])
rep3 = Classifier(classname="weka.classifiers.trees.REPTree", options=["-L", "3"])

dt1 = DecisionTreeClassifier(splitter='random', max_depth=1)
dt2 = DecisionTreeClassifier(splitter='random',max_depth=2)
dt3 = DecisionTreeClassifier(splitter='random',max_depth=2)

loader = Loader(classname="weka.core.converters.ArffLoader")

classifiers_weka = [rt1, rt2, rt3]
classifiers_sklearn = [dt1, dt2, dt3]

for file in files:
	if not file=="./datasets/bridges/bridges.arff" and not file=="./datasets/echocardiogram/echocardiogram.arff":
		dataset = load_arff(file)

		Y = dataset["target"]
		X = dataset.drop("target", axis=1)
		#if X.isnull().values.any():
		X = _preprocess_data(X)

		dt1.fit(X,Y)
		y = dt1.predict(X)
		sklearn_kappa[file] = cohen_kappa_score(Y, y)

		data = loader.load_file(file)
		data.class_is_last()

		pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
		evl = Evaluation(data)
		evl.crossvalidate_model(rt1, data, 10, Random(1), pout)
		weka_kappa[file]=(evl.kappa)
		print(sklearn_kappa[file], end='\t')
		print(weka_kappa[file])

jvm.stop()


