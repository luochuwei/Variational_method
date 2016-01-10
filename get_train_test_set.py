#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 05/01/2016
#    Usage: get train set and test set
#
############################################
import cPickle
# f = open(r'vocab.txt')
# # fw = open(r'vocab_only_word.txt','w')

# word_dic = {}
# for i,j in enumerate(f):
#     s = j.split('\t')
#     assert len(s) == 2
#     fw.write(s[0])
#     fw.write('\n')
# fw.close()
# f.close()


# def get_num(l,d):
#     a = []
#     for i in l.split():
#         if i in d:
#             a.append(d[i])
#         else:
#             a.append(d["<unk>"])

#     tag = (float(a.count(1))/float(len(a)))
#     if tag >= 0.3:
#         # print tag
#         return 0
#     else:
#         s = ''
#         for j in a:
#             s+=str(j)
#             s+=' '
#         return s[:-1]

f = open(r'vocab_only_word.txt')

word_dic = {}

for i,j in enumerate(f):
    word_dic[j[:-1]] = i
f.flush()
f.close()

cPickle.dump(word_dic, open(r'word_dic.pkl', 'wb'))

# ff = open(r'196366.txt')
# fw = open(r'196366_num.txt', 'w')

# for kk,line in enumerate(ff):
#     s = line[:-1].split('\t')
#     assert len(s) == 3
#     s0 = get_num(s[0], word_dic)
#     s1 = get_num(s[1], word_dic)
#     s2 = get_num(s[2], word_dic)
#     # break
#     if s0 != 0 and s1!=0 and s2!=0 and len(s0.split())<=70 :
#         fw.write(s0)
#         fw.write('\t')
#         fw.write(s1)
#         fw.write('\t')
#         fw.write(s2)
#         fw.write('\n')
#     # else:
#     #     print kk,' line is too much unknown'
# ff.close()
# fw.close()


# fnum = open(r'157435_num.txt')
# a = []
# for line in fnum:
#     s = line[:-1].split('\t')
#     assert len(s) == 3
#     a.append(len(s[0].split()))
# fnum.flush()
# fnum.close()

# n=0
# for i in a:
#     if i>=n:
#         n=i

# fnum = open(r'157435_num.txt')
# x = []
# y0 = []
# y1 = []
# for line in fnum:
#     item0, item1, item2 = line[:-1].split('\t')
#     x.append(item0)
#     y0.append(item1)
#     y1.append(item2)
# fnum.flush()
# fnum.close()

# x_train, x_test = x[:157000], x[157000:]
# y0_train, y0_test = y0[:157000], y0[157000:]
# y1_train, y1_test = y1[:157000], y1[157000:]



# cPickle.dump(x_train,open(r'x_train.pkl','wb'))
# cPickle.dump(x_test,open(r'x_test.pkl','wb'))

# cPickle.dump(y0_train,open(r'y0_train.pkl','wb'))
# cPickle.dump(y0_test,open(r'y0_test.pkl','wb'))

# cPickle.dump(y1_train,open(r'y1_train.pkl','wb'))
# cPickle.dump(y1_test,open(r'y1_test.pkl','wb'))
