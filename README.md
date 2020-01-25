# MLP-and-RBF-Network

## Introduction: 

In this project we will implement neural networks to solve a regression problem. Our goal is that of reconstructing in the region [−2,2] × [−1,1] a two dimensional function F : R2 → R. We do not have the analytic expression of the function, only a data set obtained by randomly sampling 300 points. We split the data set in two parts: 85% of the points were taken for the training set and 15% for the test set. We are going to 1870543 as a seed number.

## Question 1.1        

In this part, we had to implement a shallow MLP with the hyperbolic tangent as activation function.  To train our MLP, we used the two-phase training procedure. We decided to use the quasi-Newton class of methods, to compute the minimization, in which we could choose between the BFGS and the limited memory version of BFGS (L-BFGS). Since our problem wasn’t a large-scale problem, we used the BFGS, which has proved to have good performance for non-smooth optimization problems.  We left the default settings for the BFGS.

## Question 1.2 

In this section, we had to implement a generalized RBF network with supervised selection of centres and weights. In this case, we used the Gaussian as the activation function. To train our RBF, we used the same approach as for question 1.1.Due to the same reasons as before, we computed the optimization using the BFGS algorithm and leaving the default settings.

## Question 2.1 

In this part the parameters are fixed as in question 1.1, since it was required from the assignment.The aim is to train the shallow MLP with an extreme learning procedure, namely that fix randomly the values of w, b.

## Question 2.2 

In this part the parameters are fixed as in question 1.2.  Here the aim was to implement a generalized RBF network with supervised selection of weights and unsupervised selection of centres. As before the activation function is the Gaussian. We fixed the centers by choosing them randomly from the training set. We also set a range and iterate the algorithm for 50 times. 

## Question 3.1 

In this part we chose to implement the two blocks decomposition method for the shallow MLP.  The parameters are fixed as in question 1.1. The two-block decomposition method alternates the convex minimization with respect to the output weights v and the non-convex minimization with respect to w, b.
