import re
import random

f = open('C:/Users/junwonseo95/Desktop/dataset/EVAL/audio_paths2.txt', 'r', encoding='cp949')
f2 = open('C:/Users/junwonseo95/Desktop/dataset/EVAL/true_transcripts2.txt', 'r', encoding='cp949')

f3 = open('C:/Users/junwonseo95/Desktop/dataset/EVAL/audio_paths.txt', 'w')
f4 = open('C:/Users/junwonseo95/Desktop/dataset/EVAL/true_transcripts.txt', 'w')

lines = f.readlines()
lines2 = f2.readlines()



randomLines = random.sample(list(zip(lines, lines2)), 1300)

for randomLine in randomLines:
    f3.write(randomLine[0])
    f4.write(randomLine[1])
