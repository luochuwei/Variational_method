#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 28/12/2015
#    Usage: Variational Method2
#
############################################

import numpy as np
import theano
import theano.tensor as T

import cPickle as pickle
from collections import OrderedDict
import time
# theano.config.compute_test_value = 'warn'

class VM2:
    """This class implements the Variational method 2"""
    def __init__(self, total_class_num, hidden_units_encoder, hidden_units_decoder, features, latent_variables, b1, b2, learning_rate, sigma_init, batch_size):
        # theano.config.floatX = 'float32'
        self.total_class_num = total_class_num
        self.batch_size = batch_size
        self.hidden_units_encoder = hidden_units_encoder
        self.hidden_units_decoder = hidden_units_decoder
        self.features = features  #word embedding lenth
        self.latent_variables = latent_variables
        # self.continuous = continuous

        self.b1 = theano.shared(np.array(b1).astype(theano.config.floatX), name = "b1")
        self.b2 = theano.shared(np.array(b2).astype(theano.config.floatX), name = "b2")
        self.learning_rate = theano.shared(np.array(learning_rate).astype(theano.config.floatX), name="learning_rate")

        #Initialize all variables as shared variables so model can be run on GPU

        #encoder
        W_xhe = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder,features)).astype(theano.config.floatX), name='W_xhe')
        W_hhe = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder,hidden_units_encoder)).astype(theano.config.floatX), name='W_hhe')
        
        b_he = theano.shared(np.zeros((hidden_units_encoder,1)).astype(theano.config.floatX), name='b_he', broadcastable=(False,True))

        encoder_para = {}
        for i in xrange(self.total_class_num):    
            encoder_para["W_hmu_"+str(i)] = theano.shared(np.random.normal(0,sigma_init,(latent_variables,hidden_units_encoder)).astype(theano.config.floatX), name='W_hmu_'+str(i))
            encoder_para["b_hmu_"+str(i)] = theano.shared(np.zeros((latent_variables,1)).astype(theano.config.floatX), name='b_hmu_'+str(i), broadcastable=(False,True))

            encoder_para["W_hsigma_"+str(i)] = theano.shared(np.random.normal(0,sigma_init,(latent_variables,hidden_units_encoder)).astype(theano.config.floatX), name='W_hsigma_'+str(i))
            encoder_para["b_hsigma_"+str(i)] = theano.shared(np.zeros((latent_variables,1)).astype(theano.config.floatX), name='b_hsigma_'+str(i), broadcastable=(False,True))
        #decoder
        decoder_para = {}
        for j in xrange(self.total_class_num):
            decoder_para["W_Fh_"+str(j)] = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,latent_variables)).astype(theano.config.floatX), name='W_Fh_'+str(j))
            decoder_para["b_Fh_"+str(j)] = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_Fh_'+str(j), broadcastable=(False,True))

            decoder_para["W_hhd_"+str(j)] = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,hidden_units_decoder)).astype(theano.config.floatX), name='W_hhd_'+str(j))
            decoder_para["W_xhd_"+str(j)] = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,features)).astype(theano.config.floatX), name='W_xhd_'+str(j))

            decoder_para["b_hd_"+str(j)] = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_hd_'+str(j), broadcastable=(False,True))

            decoder_para["W_hx_"+str(j)] = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_decoder)).astype(theano.config.floatX), name='W_hx_'+str(j))
            decoder_para["b_hx_"+str(j)] = theano.shared(np.zeros((features,1)).astype(theano.config.floatX), name='b_hx_'+str(j), broadcastable=(False,True))
            # if self.continuous:


        W_zh = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,latent_variables)).astype(theano.config.floatX), name='W_zh')
        b_zh = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_zh', broadcastable=(False,True))

        W_hhd = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,hidden_units_decoder)).astype(theano.config.floatX), name='W_hhd')
        W_xhd = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,features)).astype(theano.config.floatX), name='W_hxd')
        
        b_hd = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_hxd', broadcastable=(False,True))
        
        W_hx = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_decoder)).astype(theano.config.floatX), name='W_hx')
        b_hx = theano.shared(np.zeros((features,1)).astype(theano.config.floatX), name='b_hx', broadcastable=(False,True))

        para_list = [("W_xhe", W_xhe), ("W_hhe", W_hhe), ("b_he", b_he),("W_zh", W_zh), ("b_zh", b_zh), ("W_hhd", W_hhd), ("W_xhd", W_xhd), ("b_hd", b_hd),
            ("W_hx", W_hx), ("b_hx", b_hx)]
        for e_p in encoder_para:
            para_list.append((e_p, encoder_para[e_p]))
        for d_p in decoder_para:
            para_list.append((d_p, decoder_para[d_p]))

        self.params = OrderedDict(para_list)

        #Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key,value in self.params.items():
            if 'b' in key:
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key, broadcastable=(False,True))
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key, broadcastable=(False,True))
            else:
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)

    def create_gradientfunctions(self):
        """This function takes as input the whole dataset and creates the entire model"""
        # theano.config.floatX = 'float32'
        def encodingstep(x_t, h_t):
            # if self.continuous:
                # return T.nnet.softplus(self.params["W_xhe"].dot(x_t) + self.params["W_hhe"].dot(h_t) + self.params["b_he"])
            # else:
            return T.tanh(self.params["W_xhe"].dot(x_t) + self.params["W_hhe"].dot(h_t) + self.params["b_he"])

        x = T.tensor3("x")
        # x.tag.test_value = np.random.rand(3, 5, 10)

        h0_enc = T.matrix("h0_enc")
        result, _ = theano.scan(encodingstep, 
                sequences = x, 
                outputs_info = h0_enc)

        h_encoder = result[-1]
        #log sigma encoder is squared
        mu_encoder = {}
        log_sigma_encoder = {}
        logpz = {}
        for i in xrange(self.total_class_num):
            mu_encoder[i] = T.dot(self.params["W_hmu_"+str(i)],h_encoder) + self.params["b_hmu_"+str(i)]
            log_sigma_encoder[i] = T.dot(self.params["W_hsigma_"+str(i)],h_encoder) + self.params["b_hsigma_"+str(i)]
            logpz[i] = 0.005 * T.sum(1 + log_sigma_encoder[i] - mu_encoder[i]**2 - T.exp(log_sigma_encoder[i]), axis = 0)

        seed = 42
        
        if "gpu" in theano.config.device:
            print 'gpu'
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
            print 'cpu'
            srng = T.shared_randomstreams.RandomStreams(seed=seed)

        #Reparametrize F

        F = {}
        for j in xrange(self.total_class_num):
            eps = srng.normal((self.latent_variables,self.batch_size), avg = 0.0, std = 1.0, dtype=theano.config.floatX)
            F[j] = mu_encoder[j] + T.exp(0.5 * log_sigma_encoder[j]) * eps

        #calculate Z
        z_mu = mu_encoder[0]
        z_log_sigma = log_sigma_encoder[0]
        for k in xrange(1, self.total_class_num):
            z_mu+=mu_encoder[k]
            z_log_sigma += log_sigma_encoder[k]
        logP_Z = 0.005 * T.sum(1 + z_log_sigma - z_mu**2 - T.exp(z_log_sigma), axis = 0)
        eps = srng.normal((self.latent_variables,self.batch_size), avg = 0.0, std = 1.0, dtype=theano.config.floatX)
        Z = z_mu + T.exp(0.5 * z_log_sigma) * eps

        h0_dec_Z = T.tanh(self.params["W_zh"].dot(Z) + self.params["b_zh"])
        h0_dec_F = {}
        for ii in xrange(self.total_class_num):
            h0_dec_F[ii] = T.tanh(self.params["W_Fh_"+str(ii)].dot(F[ii]) + self.params["b_Fh_"+str(ii)])

        def decodingstep_z(x_t, h_t):
            h = T.tanh(self.params["W_hhd"].dot(h_t) + self.params["W_xhd"].dot(x_t) + self.params["b_hd"])
            x = T.nnet.sigmoid(self.params["W_hx"].dot(h) + self.params["b_hx"])
            return x, h
            

        x0 = T.matrix("x0")
        [y, _], _ = theano.scan(decodingstep_z,
                n_steps = x.shape[0], 
                outputs_info = [x0, h0_dec_Z])
        # Clip y to avoid NaNs, necessary when lowerbound goes to 0
        y = T.clip(y, 1e-6, 1 - 1e-6)
        logpxz = T.sum(-T.nnet.binary_crossentropy(y,x), axis = 1)

        logpxz = T.mean(logpxz, axis = 0)

        x0_F = {}
        y_F = {}
        response_y = {0:0,1:1}
        logpyf = {}

        for jj in xrange(self.total_class_num):
            def decodingstep_F(x_t, h_t):
                h = T.tanh(self.params["W_hhd_"+str(jj)].dot(h_t) + self.params["W_xhd_"+str(jj)].dot(x_t) + self.params["b_hd_"+str(jj)])
                x = T.nnet.sigmoid(self.params["W_hx_"+str(jj)].dot(h) + self.params["b_hx_"+str(jj)])
                return x, h
            x0_F[jj] = T.matrix()
            response_y[jj] = T.tensor3()
            # response_y[jj].tag.test_value = np.random.rand(3, 5, 10)
            [y_F[jj], _], _ = theano.scan(decodingstep_F,
                n_steps = response_y[jj].shape[0], 
                outputs_info = [x0_F[jj], h0_dec_F[jj]])
            y_F[jj] = T.clip(y_F[jj], 1e-6, 1 - 1e-6)
            logpyf[jj] = T.sum(-T.nnet.binary_crossentropy(y_F[jj],response_y[jj]), axis = 1)
            logpyf[jj] = T.mean(logpyf[jj], axis = 0)
        # print '~~~~~~~~~~~~~~~~~~~~~~~~~'
        #lowerbound
        logpx = logpxz + logP_Z
        for jjj in xrange(self.total_class_num):
            logpx += (logpyf[jjj] + logpz[jjj])

        #Average over time dimension
        # logpx = T.clip(logpx, 1e-30, 1 - 1e-30)
        logpx = T.mean(logpx)

        #compute all the gradients
        gradients = T.grad(logpx, self.params.values())
        self.gradients = gradients
        # print gradients
        #Let Theano handle the updates on parameters for speed
        updates = OrderedDict()
        # epoch = T.iscalar("epoch")
        epoch = T.fscalar("epoch")
        gamma = T.sqrt(1 - (1 - self.b2)**epoch)/(1 - (1 - self.b1)**epoch)

        #Adam
        for parameter, gradient, m, v in zip(self.params.values(), gradients, self.m.values(), self.v.values()):
            new_m = self.b1 * gradient + (1 - self.b1) * m
            new_v = self.b2 * (gradient**2) + (1 - self.b2) * v

            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v)+ 1e-8)
            updates[m] = new_m
            updates[v] = new_v
        # print updates
        # batch = T.iscalar('batch')
        givens = {
            h0_enc: np.zeros((self.hidden_units_encoder,self.batch_size)).astype(theano.config.floatX), 
            x0:     np.zeros((self.features,self.batch_size)).astype(theano.config.floatX),
            # x:      data[0].astype(theano.config.floatX),
            x0_F[0]:     np.zeros((self.features,self.batch_size)).astype(theano.config.floatX),
            x0_F[1]:    np.zeros((self.features,self.batch_size)).astype(theano.config.floatX)
            # response_y[0]:    data[1].astype(theano.config.floatX),
            # response_y[1]:    data[2].astype(theano.config.floatX)
        }

        self.updatefunction = theano.function([x, response_y[0],response_y[1],epoch], logpx, updates=updates, givens=givens, allow_input_downcast=False)

        return True

    def encode(self, x):
        """Helper function to compute the encoding of a datapoint to latent_variables"""
        h = np.zeros((self.hidden_units_encoder,1))

        W_xhe = self.params["W_xhe"].get_value()
        W_hhe = self.params["W_hhe"].get_value()
        b_hhe = self.params["b_he"].get_value()

        encoder_para = {}
        for i in xrange(self.total_class_num):    
            encoder_para["W_hmu_"+str(i)] = self.params["W_hmu_"+str(i)].get_value()
            encoder_para["b_hmu_"+str(i)] = self.params["b_hmu_"+str(i)].get_value()

            encoder_para["W_hsigma_"+str(i)] = self.params["W_hsigma_"+str(i)].get_value()
            encoder_para["b_hsigma_"+str(i)] = self.params["b_hsigma_"+str(i)].get_value()
        for t in xrange(x.shape[0]):
            h = np.tanh(W_xhe.dot(x[t]) + W_hhe.dot(h) +b_hhe)
            # print h
        
        mu_encoder = {}
        log_sigma_encoder = {}
        F = {}

        for j in xrange(self.total_class_num):
            mu_encoder[j] = encoder_para["W_hmu_"+str(j)].dot(h) + encoder_para["b_hmu_"+str(j)]
            log_sigma_encoder[j] = encoder_para["W_hsigma_"+str(j)].dot(h) + encoder_para["b_hsigma_"+str(j)]
            F[j] = np.random.normal(mu_encoder[j], np.exp(log_sigma_encoder[j]))
            if j == 0:
                z_mu = mu_encoder[j]
                z_log_sigma = log_sigma_encoder[j]
            else :
                z_mu += mu_encoder[j]
                z_log_sigma += log_sigma_encoder[j]
        z = np.random.normal(z_mu, np.exp(z_log_sigma))
        # z = (z_mu, np.exp(z_log_sigma))

        return F, z

    def decode_z(self, t_steps, z):
        """
        Helper function to compute the decoding of a datapoint from z to x and F[i] to response_y[i]
        t_steps starts from 1
        if tag == 0     we do not decode z to x
        else if tag == 1 we decode z to x
        """

        # x = np.zeros((t_steps+1, self.features))
        # xx = [[] for i in range(t_steps)]
        x = [[] for i in xrange(t_steps+1)]
        x[0] = np.zeros((self.features, z.shape[1]))


        W_zh = self.params['W_zh'].get_value()
        b_zh = self.params['b_zh'].get_value()

        W_hhd = self.params['W_hhd'].get_value()
        W_xhd = self.params['W_xhd'].get_value()
            
        b_hd = self.params['b_hd'].get_value()
            
        W_hx = self.params['W_hx'].get_value()
        b_hx = self.params['b_hx'].get_value()

        h = W_zh.dot(z) + b_zh

        for t in xrange(t_steps):
            # h = np.tanh(W_hhd.dot(h) + W_xhd.dot(x[t,:,np.newaxis]) + b_hd)
            h = np.tanh(W_hhd.dot(h) + W_xhd.dot(x[t]) + b_hd)
            # print h
            # x[t+1, :] = np.squeeze(1 /(1 + np.exp(-(W_hx.dot(h) + b_hx))))
            # x[t+1, :] = 1 /(1 + np.exp(-(W_hx.dot(h) + b_hx)))
            # x[t+1] = np.squeeze(1 /(1 + np.exp(-(W_hx.dot(h) + b_hx))))
            # xx[t]=W_hx.dot(h) + b_hx
            x[t+1] = 1 /(1 + np.exp(-(W_hx.dot(h) + b_hx)))

        return np.array(x[1:])
        # return np.array(xx)

    def decode_F(self, tag, F, type_n, y_t, h_t):
        """
        F is latent_variables
        Helper function to compute the decoding of a datapoint from F to response_y
        type_n is the type number of response_y (0,1,.....)
        if tag is 1, then it decode the first word
        if tag is 0, then dosen't decode the first word
        """
        # response_y = np.zeros((t_steps+1, self.features))


        W_Fh = self.params["W_Fh_"+str(type_n)].get_value()
        b_Fh = self.params["b_Fh_"+str(type_n)].get_value()

        W_hhd = self.params["W_hhd_"+str(type_n)].get_value()
        W_xhd = self.params["W_xhd_"+str(type_n)].get_value()

        b_hd = self.params["b_hd_"+str(type_n)].get_value()

        W_hx = self.params["W_hx_"+str(type_n)].get_value()
        b_hx = self.params["b_hx_"+str(type_n)].get_value()

        if tag == 1:
            h = W_Fh.dot(F) + b_Fh
            y = np.zeros((self.features, F.shape[1]))
            h = np.tanh(W_hhd.dot(h) + W_xhd.dot(x[t]) + b_hd)
            y0 = 1 /(1 + np.exp(-(W_hx.dot(h) + b_hx)))
            return y0, h
        elif tag == 0:
            h = np.tanh(W_hhd.dot(h_t) + W_xhd.dot(y_t) + b_hd)
            y_t_1 = 1 /(1 + np.exp(-(W_hx.dot(h) + b_hx)))
            return y_t_1, h

    def save_parameters(self, path):
        """Saves all the parameters in a way they can be retrieved later"""
        pickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/params.pkl", "wb"))
        pickle.dump({name: m.get_value() for name, m in self.m.items()}, open(path + "/m.pkl", "wb"))
        pickle.dump({name: v.get_value() for name, v in self.v.items()}, open(path + "/v.pkl", "wb"))

    def save_batch_parameters(self, path, epoch, i):
        """Saves all the parameters in a way they can be retrieved later"""
        pickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/" +str(epoch)+"_"+str(i) + "params.pkl", "wb"))
        # pickle.dump({name: m.get_value() for name, m in self.m.items()}, open(path + "/" + str(epoch)+"_"+str(i) + "m.pkl", "wb"))
        # pickle.dump({name: v.get_value() for name, v in self.v.items()}, open(path + "/" + str(epoch)+"_"+str(i) + "v.pkl", "wb"))

    def load_parameters(self, path):
        """Load the variables in a shared variable safe way"""
        p_list = pickle.load(open(path + "/params.pkl", "rb"))
        m_list = pickle.load(open(path + "/m.pkl", "rb"))
        v_list = pickle.load(open(path + "/v.pkl", "rb"))

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))





# Zdim = 200
# hidden_units_encoder = 300
# hidden_units_decoder = 300
# features = 26111
# batch_size = 5

# a = np.random.random_integers(0,1,(30,features,200)).astype(theano.config.floatX)
# b = np.random.random_integers(0,1,(30,features,200)).astype(theano.config.floatX)
# c = np.random.random_integers(0,1,(30,features,200)).astype(theano.config.floatX)

# data = [a,b,c]

# v = VM2(2,hidden_units_encoder,hidden_units_decoder,features,Zdim,0.05,0.001,0.005,0.1,batch_size)
# print 'create_gradientfunctions'
# v.create_gradientfunctions()

# print 'update'
# # # l=v.updatefunction(a[:,:,0:2],b[:,:,0:2],c[:,:,0:2],float(2))
# # l=v.updatefunction(a[:,:,0:batch_size],b[:,:,0:batch_size],c[:,:,0:batch_size],float(2))

# # print 'save parameters'
# # v.save_parameters(r'E:\Learning\VAE\idea\code\para')

# best_lower_bound = -np.inf

# for ep in xrange(1,20000):
#     all_lower_bound = 0.0
#     t1 = time.time()
#     for i in xrange(10):
#         l=v.updatefunction(a[:,:,i*batch_size:(i+1)*batch_size],b[:,:,i*batch_size:(i+1)*batch_size],c[:,:,i*batch_size:(i+1)*batch_size],float(ep))
#         all_lower_bound += l
#         # print l
#     t2 = time.time()
#     print "~~~~~~~cost : ",(t2-t1),"s ~~~~~~~~~~"
#     print "all lowerbound : ", all_lower_bound
#     # print v.params["W_xhe"].get_value()[0][0]
#     # if all_lower_bound > best_lower_bound:
#     #     print 'new best lowerbound, save parameters'
#     #     best_lower_bound = all_lower_bound
#     #     v.save_parameters(r'E:\Learning\VAE\idea\code\para')


# # print 'save_parameters'
# # v.save_parameters(r'E:\Learning\VAE\idea\code\para')

# f,z = v.encode(a)

# p = v.decode_z(3,z)
