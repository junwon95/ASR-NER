import re
import Levenshtein

# PER '|' , LOC '$' , ORG '{'

def get_cer(tar, pred):

    for x in '$|{]':
        tar.replace(x,'')
        pred.replace(x,'')

    return Levenshtein.distance(tar, pred), len(tar)

def get_ne_cer(tar, pred):

    distance = 0
    length = 0

    for t, p in zip(tar, pred):
        t = t.replace(' ', '')
        p = p.replace(' ', '')

    if len(tar) == len(pred):
        for t,p in zip(tar, pred):
            distance += Levenshtein.distance(t,p)
            length += len(t)

    elif len(tar) < len(pred):
        for t in tar:
            distance += min(map(lambda x: Levenshtein.distance(t,x), pred))
            length += len(t)

    elif len(tar) > len(pred):
        for p in pred:
            candidates =list(map(lambda x: Levenshtein.distance(p,x), tar))
            optimal = min(candidates)
            distance += optimal
            length += len(candidates.index(optimal))

    return distance, length





#---------- read data -----------
f = open("TEST/true_transcripts.txt", 'rt', encoding="UTF8")
f2 = open("OUTPUTS/NER-OUT/final_output.txt", 'rt', encoding="UTF8")

targets = f.readlines()
predictions = f2.readlines()

#---------- statistics ----------
total_distance = 0
total_length = 0

ne_distance = 0
ne_length = 0

per_distance = 0
per_length = 0

loc_distance = 0
loc_length = 0

org_distance = 0
org_length = 0

per_cnt = 0
loc_cnt = 0
org_cnt = 0

for target, prediction in zip(targets,predictions):

    PER = re.findall('\|.*?]', target)
    LOC = re.findall('\$.*?]', target)
    ORG = re.findall('\{.*?]', target)
    PER()

    P_PER = re.findall('\|.*?]', prediction)
    P_LOC = re.findall('\$.*?]', prediction)
    P_ORG = re.findall('\{.*?]', prediction)

    dist, length = get_cer(target, prediction)
    total_distance += dist
    total_length += length

    dist, length = get_ne_cer(PER+LOC+ORG, P_PER+P_LOC+P_ORG)
    ne_distance += dist
    ne_length += length

    dist, length = get_ne_cer(PER, P_PER)
    per_distance += dist
    per_length += length

    dist, length = get_ne_cer(LOC, P_LOC)
    loc_distance += dist
    loc_length += length

    dist, length = get_ne_cer(ORG, P_ORG)
    org_distance += dist
    org_length += length

    per_cnt += len(PER)
    loc_cnt += len(LOC)
    org_cnt += len(ORG)



print('------------------TEST RESULTS-------------------')
print('validation set size: ' + str(len(targets)))
print('Character Error Rates')
print('total CER: ' + str(total_distance/total_length))
print('named-entity CER: ' + str(ne_distance/ne_length))

print('PER tags: ' + str(per_cnt))
if per_cnt:
    print('PER tag CER: ' + str(per_distance/per_length))

print('LOC tags: ' + str(loc_cnt))
if loc_cnt > 0:
    print(loc_cnt)
    print('LOC tag CER: ' + str(loc_distance/loc_length))

print('ORG tags: ' + str(org_cnt))
if org_cnt:
    print('ORG tag CER: ' + str(org_distance/org_length))
