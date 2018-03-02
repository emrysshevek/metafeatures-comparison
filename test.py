import scipy as sp
from scipy.stats import bayes_mvs, ttest_1samp
import numpy as np
import csv
import pandas as pd


#input scores
f_scores = pd.read_csv('./f_scores.csv', sep=',')
speed_scores = pd.read_csv('./speed_scores.csv',sep=',')
cod_scores = pd.read_csv('./cod_scores.csv',sep=',')

#compute confidence intervals for means
f_mean, f_var, f_std = bayes_mvs(data=f_scores,alpha=.95)
speed_mean, speed_var, speed_std = bayes_mvs(data=speed_scores,alpha=.95)
cod_mean, cod_var, cod_std = bayes_mvs(data=cod_scores,alpha=.95)

#perform 1 sided t test for means
f_prob = ttest_1samp(f_scores,0)
speed_prob = ttest_1samp(speed_scores,0)
cod_prob = ttest_1samp(cod_scores,0)

print("F Score Probability: " + str(f_prob))
print("F Score mean: " + str(f_mean))
print()

print("Speed Score Probability: " + str(speed_prob))
print("Speed Score mean: " + str(speed_mean))
print()

print('COD Score Probability: ' + str(cod_prob))
print("COD Score mean: " + str(cod_mean))
print()
