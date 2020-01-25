import numpy as np
import pandas as pd
import functions_question_2_1 as f


import warnings
warnings.filterwarnings("ignore")

np.random.seed(1870543)

neurons = 24
sigma = 2.4
rho = 10e-5

X_tr, y_tr, X_ts, y_ts = f.data_set()
test_error, train_error, train_time, w, v, b, FE, GE, NI, error, terror = f.XLearn(X_tr, y_tr, X_ts, y_ts, neurons, rho, sigma, 2, 1, 50)

print("\nTraining Finished")

f.writejson("model_2_1.json", w, b, v, neurons, sigma, rho)

f.plotting('graph_question_2_1.png',w,b,v,sigma)

f.writetxt('output_question_2_1.txt', w,b,v, neurons, sigma, rho, 'BFGS', train_error, test_error, train_time, FE, GE, NI)
