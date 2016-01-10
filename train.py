import numpy as np
# import theano
# import theano.tensor as T

import cPickle
# from collections import OrderedDict
import time

import Variational_Method2

#x = ['1 2 3 4 0','2 4 0 1', '0 2 1 3 3','4 4 3','2 1 3 3 3']

def constuct_batch_data(batch, batch_size, features, x):
    # return 1
    batch_x = x[batch*batch_size:(batch+1)*batch_size]
    max_len_batch_x = max([len(i.split()) for i in batch_x])
    # max_len_batch_x = 70
    # print max_len_batch_x

    for j in xrange(batch_size):
        j_len = len(batch_x[j].split())
        # print j_len
        if j_len < max_len_batch_x:
            for k in xrange(max_len_batch_x - j_len):
                batch_x[j] += ' 0'
    
    final_x = []
    for jj in xrange(max_len_batch_x):
        batch_word_vec = []
        for kk in xrange(batch_size):
            v = np.zeros(features)
            v[int(batch_x[kk].split()[jj])] = 1
            batch_word_vec.append(v)
        batch_word_vec = np.array(batch_word_vec).transpose()
        final_x.append(batch_word_vec)
    return np.array(final_x).astype('float32')


Zdim = 200
hidden_units_encoder = 300
hidden_units_decoder = 300
features = 26111
batch_size = 2


x = cPickle.load(open(r'E:\Learning\VAE\idea\code\data\x_train.pkl','rb'))

y0 = cPickle.load(open(r'E:\Learning\VAE\idea\code\data\y0_train.pkl','rb'))

y1 = cPickle.load(open(r'E:\Learning\VAE\idea\code\data\y1_train.pkl','rb'))



v = Variational_Method2.VM2(2,hidden_units_encoder,hidden_units_decoder,features,Zdim,0.05,0.001,0.005,0.1,batch_size)
print 'create_gradientfunctions'
v.create_gradientfunctions()

print 'update'

best_lower_bound = -np.inf
scan_times = len(x)/batch_size

for ep in xrange(1,20000):
    # all_lower_bound = 0.0
    lower_bound_batch_data = 0.0
    lower_bound_1000_batch = 0.0 
    t1 = time.time()
    for i in xrange(scan_times):
        l=v.updatefunction(constuct_batch_data(i, batch_size, features, x), constuct_batch_data(i, batch_size, features, y0), constuct_batch_data(i, batch_size, features, y1), float(ep))

        lower_bound_batch_data += l
        lower_bound_1000_batch += l

        if (i+1)%100 == 0:
            t2 = time.time()
            print "~~~~~~~200 cost time : ",(t2-t1),"s ~~~~~~~~~~"
            t1 = time.time()
            print "epoch : ", ep," scan of ",i+1,"/",scan_times," ",100*(i+1)/float(scan_times),"% >>> 200 data lowerbound : ", lower_bound_batch_data
            lower_bound_batch_data = 0.0

        if (i+1)%1000 == 0:
            if lower_bound_1000_batch > best_lower_bound:
                best_lower_bound = lower_bound_1000_batch
                lower_bound_1000_batch = 0.0
                v.save_parameters(r'E:\Learning\VAE\idea\code\para')
            else:
                lower_bound_1000_batch = 0.0

        if (i+1)%5000 == 0:
            #do not save m and v, only save 
            v.save_batch_parameters(r'E:\Learning\VAE\idea\code\para',ep,i)
