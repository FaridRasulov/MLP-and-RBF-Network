import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions_question_1_1 as f

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1870543)

input_dim = 2
output_dim = 1
N = [i for i in range(8,25,4)]
S = [2.4]#[1,1.5,2,2.4]
R = [10e-5]#[10e-5,10e-4,10e-3]

x_train,y_train,x_test,y_test = f.data_set()
test_error, train_error, val_error, train_time, neurons, sigma, rho, w, v, b, FE, GE, NI, error, terror = f.gridsearch(x_train,y_train,x_test,y_test, input_dim,output_dim,N,S,R)

print("\nTraining Finished")

f.writejson("model_1_1.json", w, b, v, neurons, sigma, rho)
f.writetxt("output_question_1_1.txt", w,b,v, neurons, sigma, rho, 'BFGS', train_error, test_error, val_error, train_time, FE, GE, NI)
#f.ovunfit(N,S,R,sigma,rho,error,terror)
f.plotting('graph_question_1_1.png',w,b,v,sigma)