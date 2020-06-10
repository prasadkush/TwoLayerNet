import numpy as np
import random
import math


def loss_svm_i(Y_, h_, W_, delta, i, lambd):
    l = h_[i,:] - h_[i,Y_[i]] + delta
    l[Y_[i]] = 0
    l = np.maximum(0,l)
    return sum(l) 

def grad_l_w_i(Xs_, Y_, h_, W_, delta, i, lambd):
	grad_w = h_[i,:] - h_[i,Y_[i]] + delta
	grad_w = grad_w > 0
	grad_w = np.multiply(grad_w.reshape(grad_w.shape[0],1),np.tile(Xs_[i,:],(grad_w.shape[0],1)))
	grad_w[Y_[i],:] = -(sum(grad_w[0:Y_[i],:]) + sum(grad_w[Y_[i]+1:grad_w.shape[0],:]))
	grad_w = grad_w + lambd*W_.transpose()
	return grad_w

def loss_svm(Y_, h_, W_, delta, lambd, samples, s):
	sum_loss = 0

	if samples == h_.shape[0] :
		for i in range(h_.shape[0]) :
			sum_loss = sum_loss + loss_svm_i(Y_, h_, W_, delta, i, lambd)
		loss = float(sum_loss)/float(h_.shape[0])
	elif samples < h_.shape[0] and samples > 0 :
		#s = random.sample(range(0,h_.shape[0]),samples)
		for i in s :
			sum_loss = sum_loss + loss_svm_i(Y_, h_, W_, delta, i, lambd)
		loss = float(sum_loss)/float(samples)
	return loss + lambd*sum(sum(W_*W_))/2

def loss_svm_vec(Y_, h_, W_, delta, lambd, samples, s):
	if samples < h_.shape[0] :
		h_s = h_[s,:]
		Y_s = Y_[s]
	else :
		h_s = h_
		Y_s = Y_
	l = h_s[:,:] - h_s[range(h_.shape[0]),Y_s].reshape(h_s.shape[0],1) + delta
	l[range(h_s.shape[0]),Y_s] = 0
	l_ = l
	l = np.maximum(0,l)
	kinks_crossed = (l_ != l).astype("uint8")
	loss = np.mean(np.sum(l,axis=1)) + lambd*sum(sum(W_*W_))/2
	return loss, kinks_crossed

def loss_relusvm_vec(Y_, h_, W_, delta, lambd, samples, s):
	if samples < h_.shape[0] :
		h_s = np.maximum(0,h_[s,:])
		Y_s = Y_[s]
	else :
		h_s = np.maximum(0,h_)
		Y_s = Y_
	l = h_s[:,:] - h_s[range(h_.shape[0]),Y_s].reshape(h_s.shape[0],1) + delta
	l[range(h_s.shape[0]),Y_s] = 0
	l = np.maximum(0,l)
	loss = np.mean(np.sum(l,axis=1)) + lambd*sum(sum(W_*W_))/2
	return loss

def numeric_grad(Xs_, Y_, h_p, h_m, W_p, W_m, delta, lambd, eps, samples, s):
	num_grad = 0
	num_grad = float(loss_svm(Y_, h_p, W_p, delta, lambd, samples, s) - loss_svm(Y_, h_m, W_m, delta, lambd, samples, s))/float(2*eps)
	return num_grad

def numeric_grad_vec(X_, Y_, W_, h_, delta, lambd, eps, samples, s, l):
	if samples < h_.shape[0] :
		Xs_s = X_[s,:]
		Y_s = Y_[s]
	else :
		Xs_s = X_
		Y_s = Y_
	Wp = np.repeat(W_[:,:,np.newaxis],W_.shape[0],axis=2)
	Wm = np.repeat(W_[:,:,np.newaxis],W_.shape[0],axis=2)
	Wp[range(W_.shape[0]),l,range(W_.shape[0])] += eps
	Wm[range(W_.shape[0]),l,range(W_.shape[0])] -= eps
	h_s = Xs_s.dot(W_)	
	h_sd = np.repeat(h_s[:,:,np.newaxis],W_.shape[0],axis=2)
	h_sdm = np.copy(h_sd)
	h_sd[:,l,range(W_.shape[0])] = h_sd[:,l,range(W_.shape[0])] + Xs_s[:,range(W_.shape[0])]*eps
	h_sdm[:,l,range(W_.shape[0])] = h_sdm[:,l,range(W_.shape[0])] - Xs_s[:,range(W_.shape[0])]*eps
	lp = h_sd[:,:,:] - h_sd[range(h_sd.shape[0]),Y_s,:].reshape(h_sd.shape[0],1,W_.shape[0]) + delta
	lp[range(h_sd.shape[0]),Y_s,:] = 0
	lp = np.maximum(0,lp)
	Wp =Wp*Wp
	lossp = np.mean(np.sum(lp,axis=1),axis=0) + lambd*np.sum(np.sum(Wp,axis=0),axis=0)/2
	lm = h_sdm[:,:,:] - h_sdm[range(h_sdm.shape[0]),Y_s,:].reshape(h_sdm.shape[0],1,W_.shape[0]) + delta
	lm[range(h_sdm.shape[0]),Y_s,:] = 0
	lm = np.maximum(0,lm)
	Wm = Wm*Wm
	lossm = np.mean(np.sum(lm,axis=1),axis=0) + lambd*np.sum(np.sum(Wm,axis=0),axis=0)/2
	num_grad = (lossp - lossm)/float(2*eps)
	return num_grad
    

def grad_w(Xs_, Y_, h_, W_, delta_, lambd, samples, s):
	grad = np.zeros((W_.shape[1],W_.shape[0]),dtype = float)
	if samples == h_.shape[0] :
		s_ = range(h_.shape[0])
	else :
		s_ = s
	for i in s_ :
		grad = grad + grad_l_w_i(Xs_, Y_, h_, W_, delta_, i, lambd)
	grad = grad/samples
	return grad.transpose(1,0)

def grad_w_vec(Xs_, Y_, h_, W_, delta_, lambd, samples, s):
	if samples < h_.shape[0] :
		h_s = h_[s,:]
		Y_s = Y_[s]
		Xs_s = Xs_[s,:]
	else :
		h_s = h_
		Y_s = Y_
		Xs_s = Xs_
	grad_w = h_s[:,:] - h_s[range(h_s.shape[0]),Y_s].reshape(h_s.shape[0],1) + delta_
	grad_w[range(h_s.shape[0]),Y_s] = 0
	grad_w = (grad_w > 0).astype(dtype="float")
	grad_ = np.einsum('ij...,ik...->ijk',grad_w,Xs_s)
	grad_[range(h_s.shape[0]),Y_s,:] = -np.sum(grad_,axis=1)
	grad_w = np.mean(grad_,axis = 0)
	grad_w = grad_w.transpose(1,0) + lambd*W_
	return grad_w

def grad_w_vec_(X, Y, h, W, delta, lambd, samples):
	grad_h = ((h - h[range(h.shape[0]),Y].reshape(h.shape[0],1) + delta) > 0).astype("float")
	grad_h[range(h.shape[0]),Y] = 0
	grad_h[range(h.shape[0]),Y] = -np.sum(grad_h,axis = 1)
	#print 'grad_h shape: ', grad_h.shape
	grad_w = (X.transpose().dot(grad_h))/(h.shape[0]) + lambd*W
	return grad_w

#def grad_h_vec(W_, X_, Y_, delta):
#	grad_h = (np.sum(np.maximum(0,np.repeat(W_[:,:,np.newaxis], Y_.shape[0], axis = 2) - W_[:,Y_].reshape(W_.shape[0],1,Y_.shape[0])), axis = 1).transpose(1,0))/Y_.shape[0]
#	return grad_h

def grad_h_vec(W_, h_, Y_, delta):
	lscore = h_[:,:] - h_[range(h_.shape[0]),Y_].reshape(h_.shape[0],1) + delta
	lscore = (lscore > 0).astype("float")
	lscore = lscore.transpose()
	#grad_h = (np.sum(np.einsum('ijk...,jk...->ijk',np.repeat(W_[:,:,np.newaxis], Y_.shape[0], axis = 2) - W_[:,Y_].reshape(W_.shape[0],1,Y_.shape[0]), lscore), axis = 1)).transpose(1,0)/Y_.shape[0]
	W_new = np.repeat(W_[:,:,np.newaxis], Y_.shape[0], axis = 2) - W_[:,Y_].reshape(W_.shape[0],1,Y_.shape[0])
	lscore = np.repeat(lscore[np.newaxis,:,:], W_.shape[0], axis = 0)
	grad_h = (np.sum(W_new*lscore,axis=1)).transpose()/Y_.shape[0]
	return grad_h

def grad_h_vec_2(W_, h_, Y_, delta):
	lscore = h_[:,:] - h_[range(h_.shape[0]),Y_].reshape(h_.shape[0],1) + delta
	lscore = (lscore > 0).astype("uint8")
	lscore = lscore.transpose()
	grad_h = (np.sum(np.einsum('ijk...,jk...->ijk',np.repeat(W_[:,:,np.newaxis], Y_.shape[0], axis = 2) - W_[:,Y_].reshape(W_.shape[0],1,Y_.shape[0]), lscore), axis = 1)).transpose(1,0)/Y_.shape[0]
	return grad_h

def gradient_check(grad, Xs_, Y_, W_, h_, delta, lambd, eps, samples, s, l):
	grad_norm = np.sqrt(np.sum(grad*grad,axis=0)) 
	num_grad = numeric_grad_vec(Xs_, Y_, W_, h_, delta, lambd, eps, samples, s, l)
	num_grad_norm = np.sqrt(np.sum(num_grad*num_grad))
	grad_norm_l = grad_norm[l]
	print 'grad norm: ', grad_norm_l
	print 'num_grad_norm: ', num_grad_norm
	print 'grad norm diff: ', abs(grad_norm_l - num_grad_norm)

