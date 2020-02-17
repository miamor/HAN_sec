import os

words_unique = []
with open('/media/tunguyen/Devs/Security/HAN_sec/data/adniu_iapi/vocab/edge.txt', 'r') as f:
    words = f.readline().split(' ')

    for w in words:
        if w != ' ' and w not in words_unique:
            words_unique.append(w)

with open('/media/tunguyen/Devs/Security/HAN_sec/data/adniu_iapi/vocab/edge_.txt', 'w') as f:
    f.write(' '.join(words_unique))