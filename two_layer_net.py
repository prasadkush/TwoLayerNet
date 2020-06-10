import numpy as np
from scipy.misc import imread
from PIL import Image
import random
import math
from linear_svm import *
from svm_functions import *
from two_layer_functions import *
import shelve

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


#def val_err(X_, W_, Y_):
#	h = X_.dot(W_)
#	Ypred = h.argmax(axis = 1)
#	acc = np.mean((Ypred == Y_).astype(dtype="uint8"))
#	return acc

datadict = unpickle('data_batch_1')
datadict2 = unpickle('cifar-10-batches-py/data_batch_2')
datadict3 = unpickle('cifar-10-batches-py/data_batch_3')
datadict4 = unpickle('cifar-10-batches-py/data_batch_4')
datadict5 = unpickle('cifar-10-batches-py/data_batch_5')

X1 = datadict['data']
Y1 = datadict['labels']
Y1 = np.array(Y1)
X1 = np.array(X1).astype("float")

X2 = datadict2['data']
X3 = datadict3['data']
X4 = datadict4['data']
X5 = datadict5['data']


Y2 = datadict2['labels']
Y3 = datadict3['labels']
Y4 = datadict4['labels']
Y5 = datadict5['labels']

X2 = np.array(X2).astype("float")
X3 = np.array(X3).astype("float")
X4 = np.array(X4).astype("float")
X5 = np.array(X5).astype("float")

Y2 = np.array(Y2)
Y3 = np.array(Y3)
Y4 = np.array(Y4)
Y5 = np.array(Y5)

X = np.concatenate((X1,X2,X3,X4),axis=0)
Y = np.concatenate((Y1,Y2,Y3,Y4),axis=0)


print(X.shape)

datadictv = unpickle('cifar-10-batches-py/test_batch')
Xtest = datadictv['data']
Ytest = datadictv['labels']
Ytest = np.array(Ytest)
Xtest = np.array(Xtest).astype("float")

Xtest -= np.mean(Xtest,axis=0)
Xtest /= np.std(Xtest,axis=0)

Xv = X5
Yv = Y5

Xv -= np.mean(Xv,axis=0)
Xv /= np.std(Xv,axis=0)

#data preprocessing

X -= np.mean(X,axis=0)
X /= np.std(X,axis=0)

d1 = np.ones((X.shape[0],1),dtype = float)
X = np.hstack((X,d1))

d2 = np.ones((X.shape[0],1),dtype = float)

d_ = np.ones((Xv.shape[0],1),dtype = float)
Xv = np.hstack((Xv,d_))


W1 = np.random.randn(3073,100) * 0.0001/math.sqrt(3073)

W2 = np.random.randn(101,10) * 0.0001/math.sqrt(101)


print 'val acc: ', val_err(Yv,Xv,W1,W2) 

#print(val_err(Xv,W,Yv))

delta = 1.0

lambd = 0.0

learning_rate = 0.05

samples = 1

s = random.sample(range(0,X.shape[0]),X.shape[0])
X = X[s,:]
Y = Y[s]
ind_count = 0
d2_samples = np.ones((samples,1),dtype = float)

eps = 0.00001
l = 4

mu = 0.5

v_W1 = np.zeros((W1.shape[0],W1.shape[1]), dtype = float)
v_W2 = np.zeros((W2.shape[0],W2.shape[1]), dtype = float)

W1p = np.copy(W1)
W1m = np.copy(W1)

W2p = np.copy(W2)
W2m = np.copy(W2)

for t in range(400):
    print 'iteration: ', t
    # forward pass
    h1 = X[ind_count: ind_count + samples,:].dot(W1) 
    h1_ = np.hstack((h1,d2_samples))
    h1 = np.maximum(0,h1)
    h1 = np.hstack((h1,d2_samples))
    h2 = h1.dot(W2) 

    # forward pass for total number of samples
    h1f = X.dot(W1)
    h1f = np.maximum(0,h1f)
    h1f = np.hstack((h1f,d2))
    h2f = h1f.dot(W2)

    # loss computation

    loss, kinks = loss_svm_vec(Y, h2f, W2, delta, lambd, X.shape[0], s)
    loss += 0.5*lambd*sum(sum(W1))

    # backward pass

    grad_W2 = grad_w_vec(h1, Y[ind_count: ind_count + samples], h2, W2, delta, lambd, samples, s) 
    grad_W2_= grad_w_vec_(h1, Y[ind_count: ind_count + samples], h2, W2, delta, lambd, samples)
    grad_h1 = grad_h_vec(W2, h2, Y[ind_count: ind_count + samples], delta)
    grad_h1_ = grad_h1
    grad_h1 = (h1_ > 0).astype("uint8")*grad_h1
    #grad_h1_ = grad_h1
    #grad_h1 = np.maximum(0,grad_h1)
    grad_W1 = (X[ind_count: ind_count + samples,:].transpose().dot(grad_h1))
    grad_W1 = np.delete(grad_W1, W1.shape[1], axis = 1) 
    grad_W1 += lambd*W1
    #print 'grad_W1 shape: ', grad_W1.shape

    # gradient check for W1

    r = np.random.randint(0,W1.shape[0]-1)
    W1p = np.copy(W1)
    W1m = np.copy(W1)
    W1p[r,l] += eps
    W1m[r,l] -= eps
    loss_grad_w1p, kinks_crossed_p = two_layer_loss_naive(X[ind_count: ind_count + samples,:], Y[ind_count: ind_count + samples], W1p, W2, lambd, delta) 
    loss_grad_w1m, kinks_crossed_m = two_layer_loss_naive(X[ind_count: ind_count + samples,:], Y[ind_count: ind_count + samples], W1m, W2, lambd, delta)
    num_grad_w1 = (loss_grad_w1p - loss_grad_w1m)/(2*eps)
    kinks_crossed = sum(abs(kinks_crossed_p[:,l] - kinks_crossed_m[:,l])) 
    print 'kinks crossed: ', kinks_crossed
    #W1p[r,l] = W1[r,l]
    #W1m[r,l] = W1[r,l]
    rel_diff_w1 = abs(num_grad_w1 - grad_W1[r,l])/(max(abs(num_grad_w1),abs(grad_W1[r,l])) + 0.000000001)
    #if rel_diff_w1 >= 0.0000001 :
    print 'rel_diff_w1: ', rel_diff_w1
    print 'num_grad_w1: ', num_grad_w1
    print 'grad_W1: ', grad_W1[r,l]
    print 'grad_W1 shape: ', grad_W1.shape
    print ''


    # gradient check for h1

    r = np.random.randint(0,samples)
    h1p = np.copy(h1)
    h1m = np.copy(h1)
    h1p[r,l] += eps
    h1m[r,l] -= eps
    loss_grad_h1p, kinks_crossed_p = svm_loss_naive_(W2, h1p, Y[ind_count: ind_count + samples], lambd)
    loss_grad_h1m, kinks_crossed_m = svm_loss_naive_(W2, h1m, Y[ind_count: ind_count + samples], lambd)
    #loss_grad_h1p, kinks_crossed_p = loss_svm_vec(Y[ind_count:ind_count + samples], h1p, W2, delta, lambd, samples, s)
    #loss_grad_h1m, kinks_crossed_m = loss_svm_vec(Y[ind_count:ind_count + samples], h1m, W2, delta, lambd, samples, s)
    loss_grad_h1p += lambd*sum(sum(W1*W1))*0.5
    loss_grad_h1m += lambd*sum(sum(W1*W1))*0.5
    num_grad_h1 = (loss_grad_h1p - loss_grad_h1m)/(2*eps)
    kinks_crossed = sum(abs(kinks_crossed_p[r,:] - kinks_crossed_m[r,:]))
    rel_diff_h1 = abs(grad_h1_[r,l] - num_grad_h1)/(max(abs(grad_h1_[r,l]),abs(num_grad_h1)) + 0.000000001)
    print 'kinks crossed for h1: ', kinks_crossed
    #if rel_diff_h1 >= 0.0000001 :
    print 'rel_diff_h1: ', rel_diff_h1
    print 'num_grad_h1: ', num_grad_h1
    print 'grad_h1: ', grad_h1_[r,l]
    print 'grad_h1 shape: ', grad_h1.shape
    print 'r: ', r
    print ''

    # gradient check for W2
    
    r = np.random.randint(0,W2.shape[0]-1)
    W2p = np.copy(W2)
    W2m = np.copy(W2)
    W2p[r,l] += eps
    W2m[r,l] -= eps
    loss_grad_W2p, kinks_crossed_p = svm_loss_naive_(W2p, h1, Y[ind_count: ind_count + samples], lambd)
    loss_grad_W2m, kinks_crossed_m = svm_loss_naive_(W2m, h1, Y[ind_count: ind_count + samples], lambd)
    loss_grad_W2p += lambd*sum(sum(W1*W1))*0.5
    loss_grad_W2m += lambd*sum(sum(W1*W1))*0.5
    num_grad_w2 = (loss_grad_W2p - loss_grad_W2m)/(2*eps)
    rel_diff_w2 = abs(grad_W2_[r,l] - num_grad_w2)/max(abs(grad_W2_[r,l]),abs(num_grad_w2) + 0.000000001) 
    kinks_crossed = sum(abs(kinks_crossed_p[:,l] - kinks_crossed_m[:,l]))
    #W2p[r,l] = W2[r,l]
    #W2m[r,l] = W2[r,l]
    print 'kinks crossed for W2: ', kinks_crossed
    #if rel_diff_w2 >= 0.0000001 :
    print 'rel_diff_w2: ', rel_diff_w2
    print 'num_grad_w2: ', num_grad_w2
    print 'grad_w2_: ', grad_W2_[r,l]
    print 'grad_w2: ', grad_W2[r,l]
    print 'loss grad w2p: ', loss_grad_W2p
    print 'loss grad w2m: ', loss_grad_W2m
    print 'grad_W2_ shape: ', grad_W2.shape
    print 'r: ', r
    print ''
    
    #print 'W1 shape: ', W1.shape
    #print 'W2 shape: ', W2.shape


    #update_W1 = grad_W1*learning_rate
    #update_W2 = grad_W2*learning_rate

    v_W1 = mu*v_W1 - grad_W1*learning_rate
    v_W2 = mu*v_W2 - grad_W2*learning_rate

    W1 += v_W1
    W2 += v_W2

    #W1p += v_W1
    #W1m += v_W1
    #W2p += v_W2
    #W2m -= v_W2

    #W1 = W1 - update_W1
    #W2 = W2 - update_W2

    #W1p -= update_W1
    #W1m -= update_W1
    #W2p -= update_W2
    #W2m -= update_W2

    ind_count = (ind_count + samples)%(X.shape[0] - samples)

    if (t+1)%25 == 0:
        learning_rate = learning_rate/2

    print 'learning_rate: ', learning_rate
    print 'loss: ', loss
    print 'train acc: ', val_train_err(Y, h2f)
    print 'val acc: ', val_err(Yv,Xv,W1,W2) 
    print ''


    
    #optimization equation with momentum



#print 'test acc = ', val_err(Ytest,Xtest,W1,W2)

ds = shelve.open('weights_two_layer')

ds['w'] = W1

ds.close()




