import numpy as np
import pandas as pd
import functions_question_2_2 as f

import warnings
warnings.filterwarnings("ignore")
np.random.seed(1870543)
input_dim = 2
output_dim = 1
N = 12#[i for i in range(8,17,4)]
S = 1.2#[0.5,1.2,2]
R = 10e-5#[10e-5,10e-4,10e-3]

X_tr, y_tr, X_ts, y_ts = f.data_set()

test_error, train_error, train_time, n, s, r, c, v, FE, GE,NI, error, terror = f.ULearn(X_tr, y_tr, X_ts, y_ts, input_dim,output_dim,N,S,R,50)

print('MSE = {}'.format(test_error))

f.plotting('graph_question_2_2.png',c,v,s)
f.writetxt('output_question_2_2.txt', c,v, n, s, r, 'BFGS', train_error, test_error, train_time, FE, GE, NI)
f.writejson('model_2_2.json', c, v, n, s, r)