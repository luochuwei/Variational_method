#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 05/01/2016
#    Usage: get vocab
#
############################################

# f = open(r'196366.txt')
# n=1
# word_dic = {}
# # a= []
# for line in f:
#     if line.count(" EOF")!=3:
#         print n
#     s = line.split()
    
#     # a.append(line)
#     for word in s:
#         if word not in word_dic:
#             word_dic[word] = 1
#         else:
#             word_dic[word] += 1
#     n+=1



# f.close()

f = open(r'asd.txt')

word_dic = {}

for line in f:
	s = line.split('\t')
	assert len(s) == 2
	word_dic[s[0]] = int(s[1])
f.close()

sorted_x = sorted(word_dic.iteritems(), key=lambda x : x[1], reverse=True)

ff = open(r'vocab.txt','w')
for l in sorted_x:
	ff.write(l[0])
	ff.write('\t')
	ff.write(str(l[1])+'\n')
ff.close()
