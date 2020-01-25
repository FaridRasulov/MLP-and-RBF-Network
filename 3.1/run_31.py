import numpy as np
import pandas as pd
import time
import functions_question_3_1 as f

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1870543)

input_dim = 2
output_dim = 1
neurons = 24
sigma = 2.4
rho = 10e-5

w = np.random.randn(neurons, input_dim)
b = np.zeros((neurons, 1))
v = np.random.randn(1, neurons)

x_train,y_train,x_test,y_test = f.data_set()
testerror = [100]
trainerror = [100]
step = 0.003

start_time = time.time()
w, b, v, ts_error, tr_error, grad, iters = f.decomp(0, step, w, b, v, x_train, y_train, x_test, y_test, rho, sigma, input_dim, neurons, testerror, trainerror)
tr_time = time.time() - start_time

print("\nTraining Finished")

f.writetxt('output_question_3_1.txt', neurons, rho, tr_error, ts_error, tr_time, iters, grad)
f.plotting('graph_question_3_1.png',w,b,v)
f.writejson(w, b, v, neurons, rho, sigma)