import re

f = open('C:/Users/junwonseo95/Desktop/dataset/eval_clean.txt', 'r', encoding='utf8')
f2 = open('C:/Users/junwonseo95/Desktop/dataset/eval.txt', 'w')

lines = f.readlines()

for line in lines:
    path = line[:51]
    line = line[54:]

    opt = re.findall('\(.+?\)', line)
    for i, v in enumerate(opt):
        if i%2 == 0:
            line = line.replace(v+'/', '')

    for c in 'abcdefghijklmnopqrstuvwxyz/()+*':
        line = line.replace(c, '')






    f2.write(path+line)

