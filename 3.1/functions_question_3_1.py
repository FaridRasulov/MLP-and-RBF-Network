import numpy as np
import pandas as pd
from scipy.optimize import minimize,approx_fprime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import codecs,json

def data_set():
    df = pd.read_csv("DATA.csv")
    X = np.array(df[['x1','x2']])
    y = np.array(df[['y']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)
    return X_train.T, y_train.T, X_test.T, y_test.T

def wrap(w,b):
    return np.concatenate([w.reshape(-1),b.reshape(-1)])

def unwrap(inD,neu, outD,wraped):
    w = wraped[0:inD*neu].reshape(neu,inD)
    b = wraped[inD*neu:inD*neu+outD*neu].reshape(neu,outD)
    return w, b

def tanh(x,sigma):
    return (1-np.exp(-2*x*sigma))/(1+np.exp(-2*x*sigma))

def loss_function_wb(params,v,X,y,rho, sigma,input_dim,neurons):
    w,b = unwrap(input_dim,neurons,1,params)
    return (1/2)*np.mean( np.linalg.norm(DC_predict(X,w,b,v,sigma) - y,ord=2)**2 ) + rho*np.linalg.norm(params,ord=2)**2

def loss_function_v(v, w,b, X, y, rho, sigma):
    return (1/2)*np.mean( np.square(DC_predict(X,w,b,v,sigma) - y)) + rho*np.linalg.norm(v,ord=2)**2

def DC_predict(x, w, b, v, sigma):
    return np.dot(v, np.tanh(np.dot(w, x) + b))

def MSE(w,b,v,x,y,sigma):
    return 0.5*np.mean(np.square(DC_predict(x,w,b,v,sigma) - y))

def decomp(iters, step, w, b, v, x_train, y_train, x_test, y_test, rho, sigma, input_dim, neurons, ts_error, tr_error):
    step = step*0.8 if (iters > 10 and iters%5 == 0) else step
    for i in range(10):
        
        res_v = minimize(loss_function_v, x0=v, args=(w,b,x_train,y_train,rho, sigma))
        v = res_v.x.reshape((1,neurons)) 
        
        wb = wrap(w, b)
        grad = wb - step*approx_fprime(wb,loss_function_wb,0.001*np.ones_like(wb),v,x_train,y_train,rho, sigma, input_dim,neurons)
        w, b = unwrap(2, neurons, 1, grad)

    ts_error.append(MSE(w,b,v,x_test,y_test,sigma))
    tr_error.append(MSE(w,b,v,x_train,y_train,sigma))
    
    if ((ts_error[-1] < 1e-3) and (abs(ts_error[-2]-ts_error[-1]) <= 1e-3) or iters > 50):
            return w, b, v, ts_error, tr_error, grad, iters
    else: 
        return decomp(iters+1, step, w, b, v, x_train, y_train, x_test, y_test, rho, sigma, input_dim, neurons, ts_error, tr_error)
    
def writetxt(file, neurons, rho, sigma, tr_error, ts_error, tr_time, iters, grad):
    print('\nSaving results in ', file)
    output = open(file,"w")
    output.write("This is homework 1: question 3_1")
    output.write("\nNumber of neurons: " + "%i" % neurons)
    output.write("\nrho parameter: " + "%f" % rho)
    output.write("\nsigma parameter: " + "%f" % sigma)
    output.write("\nstopping criteria: Early Stopping Rule")
    output.write("\nInitial Training Error: " + "%f" % tr_error[1])
    output.write("\nFinal Training Error: " + "%f" % tr_error[-1])
    output.write("\nFinal Test Error: " + "%f" % ts_error[-1])
    output.write("\nTraining computing time: " + "%.3f" % tr_time + " sec")
    output.write("\nnumber of iterations: " + "%i" % iters)
    output.write("\nnumber of epochs: " + "%i" % 10)
    output.write("\nFunction evaluations: " + "%i" % int(iters*10))
    output.write("\nGradient evaluations: " + "%i" % int(iters*10))
    output.write("\nnorm of gradient at the optimal point: " + "%f" % np.linalg.norm(grad,ord=1))
    output.close() 

def plotting(file,w,b,v,sigma):
    print("\nPlotting the Function.....")
    X = np.linspace(-2,2,1000)
    Y = np.linspace(-2,2,1000)
    X, Y = np.meshgrid(X, Y)
    zs = DC_predict(np.array([np.ravel(X),np.ravel(Y)]),w,b,v,sigma)
    Z= zs.reshape(X.shape)
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z,linewidth=0,cmap=cm.viridis, antialiased=False)
    ax.set_xlabel('x1 Label')
    ax.set_ylabel('x2 Label')
    ax.set_zlabel('y Label')
    plt.savefig(file,dpi=600)
    plt.show()

def writejson(w, b, v, neurons, rho, sigma):
    values = {'w':w.tolist(),'b':b.tolist(),'v':v.tolist(),'neurons':neurons,'rho':rho, 'sigma': sigma}
    json.dump(values, codecs.open("model_3_1.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=4)