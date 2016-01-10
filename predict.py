import numpy as np
import theano
import theano.tensor as T

import cPickle
from collections import OrderedDict
import time

import Variational_Method2

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

def trans_to_word(v):
    batch_size = int(v.shape[-1])
    features = int(v.shape[1])

    ans = [[] for k in range(batch_size)]

    for i in range(int(v.shape[0])):
        for j in range(batch_size):
            current_array = v[:,:,j:(j+1)][i].transpose()[0]
            max_item = max(current_array)
            max_item_index = list(current_array).index(max_item)
            ans[j].append(max_item_index)

    return ans
def decode(v, t_steps, z, type_n):
        # x = np.zeros((t_steps+1, self.features))
        # xx = [[] for i in range(t_steps)]
    x = [[] for i in xrange(t_steps+1)]
    x[0] = np.zeros((v.features, z.shape[1]))


    W_Fh = v.params["W_Fh_"+str(type_n)].get_value()
    b_Fh = v.params["b_Fh_"+str(type_n)].get_value()

    W_hhd = v.params["W_hhd_"+str(type_n)].get_value()
    W_xhd = v.params["W_xhd_"+str(type_n)].get_value()

    b_hd = v.params["b_hd_"+str(type_n)].get_value()

    W_hx = v.params["W_hx_"+str(type_n)].get_value()
    b_hx = v.params["b_hx_"+str(type_n)].get_value()

    h = W_Fh.dot(z) + b_Fh

    for t in xrange(t_steps):
        h = np.tanh(W_hhd.dot(h) + W_xhd.dot(x[t]) + b_hd)
        x[t+1] = 1 /(1 + np.exp(-(W_hx.dot(h) + b_hx)))

    return np.array(x[1:])


word_dic = cPickle.load(open(r'E:\Learning\VAE\idea\code\data\word_dic.pkl', 'rb'))
num_word_dic = {}
for i,j in word_dic.iteritems():
    num_word_dic[j] = i

x = cPickle.load(open(r'E:\Learning\VAE\idea\code\data\x_test.pkl','rb'))
fwx = open('test_word.txt','w')
for jj in x:
    for kk in jj.split():
        fwx.write(num_word_dic[int(kk)])
    fwx.write('\n')
fwx.flush()
fwx.close()



Zdim = 200
hidden_units_encoder = 300
hidden_units_decoder = 300
features = 26111
batch_size = 2

print '1 build model'
v = Variational_Method2.VM2(2,hidden_units_encoder,hidden_units_decoder,features,Zdim,0.05,0.001,0.005,0.1,batch_size)
v.load_parameters(r'E:\Learning\VAE\idea\code\para')
print '2 load over'

# x_0batch = constuct_batch_data(0, batch_size, features, x)
print '3 encode'
# f,z = v.encode(x_0batch)
print '4 decode'
# y0 = decode(v,20,f[0],0)

fwww = open(r'r.txt', 'w')
for b in range(len(x)/batch_size):
    x_batch = constuct_batch_data(b, batch_size, features, x)
    f,z = v.encode(x_batch)
    zz = trans_to_word(v.decode_z(int(x_batch.shape[0]), z))
    y0 = trans_to_word(decode(v, 20, f[0], 0))
    y1 = trans_to_word(decode(v, 20, f[1], 1))
    for r in range(2):
        for l in zz[r]:
            if l!=0:
                fwww.write(num_word_dic[l])
        fwww.write('\t')
        for l in y0[r]:
            if l != 0:
                fwww.write(num_word_dic[l])
        fwww.write('\t')
        for l in y1[r]:
            if l !=0:
                fwww.write(num_word_dic[l])
        fwww.write('\n')
fwww.flush()
fwww.close()

