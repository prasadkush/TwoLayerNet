import numpy as np

def numeric_grad_W1_vec(X_, Y_, W1, W2, lambd, delta, eps, l):
    W1_p = np.repeat(W1[:,:,np.newaxis], W1.shape[0], axis = 2)
    W1_m = np.repeat(W1[:,:,np.newaxis], W1.shape[0], axis = 2)
    W1_p[range(W1.shape[0]),l,range(W1.shape[0])] += eps
    W1_m[range(W1.shape[0]),l,range(W1.shape[0])] -= eps
    h1p = np.maximum(0,np.einsum('ij...,jkl...->ikl', X_, W1_p))
    h1m = np.maximum(0,np.einsum('ij...,jkl...->ikl', X_, W1_m))
    d2 = np.ones((X_.shape[0],1,W1.shape[0])).astype("float")
    h1p = np.concatenate((h1p,d2),axis = 1)
    h1m = np.concatenate((h1m,d2),axis = 1)
    h2p = np.einsum('ijk...,jl...->ilk', h1p, W2)
    h2m = np.einsum('ijk...,jl...->ilk', h1m, W2)
    lossp = h2p[range(h2p.shape[0]),:,:] - h2p[range(h2p.shape[0]),Y_,:].reshape(h2p.shape[0],1,h2p.shape[2]) + delta
    lossm = h2m[range(h2m.shape[0]),:,:] - h2m[range(h2m.shape[0]),Y_,:].reshape(h2p.shape[0],1,h2p.shape[2]) + delta
    lossp[range(h2p.shape[0]),Y_,:] = 0
    lossm[range(h2m.shape[0]),Y_,:] = 0
    lossp = np.maximum(lossp,0)
    lossm = np.maximum(lossm,0)
    lossp = np.mean(np.sum(lossp,axis=1),axis=0)
    lossm = np.mean(np.sum(lossm,axis=1),axis=0)
    num_grad = (lossp - lossm)/2*eps
    return num_grad

def numeric_grad_W1(X_, Y_, W1, W2, lambd, delta, eps, l):
    W1_p = np.copy(W1)
    W1_m = np.copy(W1)
    W1_p[1000,l] += eps
    W1_m[1000,l] -= eps
    h1p = np.maximum(0,X_.dot(W1_p))
    h1m = np.maximum(0,X_.dot(W1_m))
    d2 = np.ones((X_.shape[0],1)).astype("float")
    h1p = np.hstack((h1p,d2))
    h1m = np.hstack((h1m,d2))
    h2p = h1p.dot(W2)
    h2m = h1m.dot(W2)
    lossp = h2p[range(h2p.shape[0]),:] - h2p[range(h2p.shape[0]),Y_].reshape(h2p.shape[0],1) + delta
    lossm = h2m[range(h2m.shape[0]),:] - h2m[range(h2m.shape[0]),Y_].reshape(h2p.shape[0],1) + delta
    lossp[range(h2p.shape[0]),Y_] = 0
    lossm[range(h2m.shape[0]),Y_] = 0
    lossp = np.maximum(lossp,0)
    lossm = np.maximum(lossm,0)
    lossp = np.mean(np.sum(lossp,axis=1),axis=0)
    lossm = np.mean(np.sum(lossm,axis=1),axis=0)
    num_grad = (lossp - lossm)/2*eps
    return num_grad

def two_layer_loss_naive(X, Y, W1, W2, lambd, delta):
    loss = 0.0
    kinks_crossed = np.zeros((X.shape[0],W1.shape[1]))
    for i in range(X.shape[0]):
        h1 = X[i,:].dot(W1)
        h1_ = h1
        h1 = np.maximum(0,h1)
        kinks_crossed[i,:] = (h1_ != h1).astype("uint8")
        d = np.array([1]).astype("float")
        h1 = np.hstack((h1,d))
        h2 = h1.dot(W2)
        correct_score = h2[Y[i]]
        for j in range(h2.shape[0]):
        	if j != Y[i] :
        		loss += np.maximum(0, h2[j] - correct_score + delta)
    loss = float(loss)/float(X.shape[0]) + lambd*( sum(sum(W1*W1)) + sum(sum(W2*W2)) )/2
    return loss, kinks_crossed

def val_train_err(Y, h2):
    Ypred = np.argmax(h2, axis=1)
    acc = np.mean((Ypred == Y).astype(dtype="uint8"))
    return acc

def val_err(Y, X, W1, W2):
    d2 = np.ones((X.shape[0],1),dtype = float)
    h1f = X.dot(W1)
    h1f = np.maximum(0,h1f)
    h1f = np.hstack((h1f,d2))
    h2f = h1f.dot(W2)
    Ypred = np.argmax(h2f, axis=1)
    acc = np.mean((Ypred == Y).astype(dtype="uint8"))
    return acc

