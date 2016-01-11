#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 23/12/2015
#    Usage: split data by cosine_similarity
#
############################################

import jieba
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.word2vec import *
import numpy as np


print 'load original data'
f = open(r'E:\Learning\conversation_dataset\Short Text Conversation\repos\repos-id-post-cn')

post = []
for i in f:
    post.append(i[:-1].split('\t'))

f.close()

f1 = open(r'E:\Learning\conversation_dataset\Short Text Conversation\repos\repos-id-cmnt-cn')
repos = []

for j in f1:
    repos.append(j[:-1].split('\t'))

f1.close()
print 'done'

print 'load word2vec model'
model = Word2Vec.load_word2vec_format(r'E:\Learning\conversation_dataset\all_word_vec.txt', binary = False)
print 'done'


print 'processing~~~'
assert len(post) == len(repos)

post_repos = {}
post_cos = {}

for k in range(len(post)):
    seg_list = ' '.join(jieba.cut(post[k][1])).split()
    s_vec = 0
    for word in seg_list:
        if word in model.vocab:
            s_vec += model[word]
    if post[k][1] not in post_repos:
    	r_seg = ' '.join(jieba.cut(repos[k][1])).split()
        r_vec = 0
        for word in r_seg:
            if word in model.vocab:
                r_vec += model[word]
        cos = cosine_similarity(s_vec, r_vec)
        post_repos[post[k][1]] = [repos[k][1], repos[k][1]]
        post_cos[post[k][1]] = [cos, cos]
    else:
        r_seg = ' '.join(jieba.cut(repos[k][1])).split()
        r_vec = 0
        for word in r_seg:
            if word in model.vocab:
                r_vec += model[word]
        cos = cosine_similarity(s_vec, r_vec)
        if cos >= post_cos[post[k][1]][0]:
            post_repos[post[k][1]][0] = repos[k][1]
            post_cos[post[k][1]][0] = cos
        elif cos <= post_cos[post[k][1]][-1]:
            post_repos[post[k][1]][-1] = repos[k][1]
            post_cos[post[k][1]][-1] = cos


print 'start to write new train file'
train_x_y1_y2 = open('train_x_y1_y2.txt', 'w')

for i,j in post_repos.iteritems():
    train_x_y1_y2.write(i)
    train_x_y1_y2.write('\t')
    train_x_y1_y2.write(j[0])
    train_x_y1_y2.write('\t')
    train_x_y1_y2.write(j[1])
    train_x_y1_y2.write('\n')
print 'done'
