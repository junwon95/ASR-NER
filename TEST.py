import re
import Levenshtein


# PER '|' , LOC '$' , ORG '{'

def get_cer(tar, pred):
    for x in '$|{]':
        tar.replace(x, '')
        pred.replace(x, '')

    return Levenshtein.distance(tar, pred), len(tar)


def get_ne_cer(tar, pred):
    distance = 0
    length = 0

    for t, p in zip(tar, pred):
        t = t.replace(' ', '')
        p = p.replace(' ', '')

    if len(tar) == len(pred):
        for t, p in zip(tar, pred):
            distance += Levenshtein.distance(t, p)
            length += len(t)

    elif len(tar) < len(pred):
        for t in tar:
            distance += min(map(lambda x: Levenshtein.distance(t, x), pred))
            length += len(t)

    elif len(tar) > len(pred):
        for p in pred:
            candidates = list(map(lambda x: Levenshtein.distance(p, x), tar))
            optimal = min(candidates)
            distance += optimal
            length += len(tar[candidates.index(optimal)])

    return distance, length


def get_f1_precision(tar, pred, tags):

    target_pos = [i for i, v in enumerate(tar) if v in tags]
    prediction_pos = [i for i, v in enumerate(pred) if v in tags]

    count = 0

    for i in prediction_pos:
        if min([abs(i-x) for x in target_pos]) <=2:
            print()
            #TODO



def printCER():
    print('\ntotal CER: {:.3f}'.format(total_distance / total_length))
    print('named-entity CER: {:.3f}'.format(ne_distance / ne_length))

    print('\nPER tags: {:d}'.format(per_cnt))
    if per_cnt > 0:
        print('PER tag CER: {:.3f}'.format(per_distance / per_length))

    print('\nLOC tags: {:d}'.format(loc_cnt))
    if loc_cnt > 0:
        print('LOC tag CER: {:.3f}'.format(loc_distance / loc_length))

    print('\nORG tags: {:d}'.format(org_cnt))
    if org_cnt > 0:
        print('ORG tag CER: {:.3f}'.format(org_distance / org_length))


def printF1():
    print()


# ---------- read data -----------
f = open("TEST/true_transcripts.txt", 'rt', encoding="UTF8")
f2 = open("OUTPUTS/NER-OUT/final_output.txt", 'rt', encoding="UTF8")

targets = f.readlines()
predictions = f2.readlines()

# ---------- statistics ----------

# CER
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

# F1
precision = 0
recall = 0

for target, prediction in zip(targets, predictions):
    # -------------- check CER ---------------
    PER = re.findall('\|.*?]', target)
    LOC = re.findall('\$.*?]', target)
    ORG = re.findall('\{.*?]', target)

    P_PER = re.findall('\|.*?]', prediction)
    P_LOC = re.findall('\$.*?]', prediction)
    P_ORG = re.findall('\{.*?]', prediction)

    dist, length = get_cer(target, prediction)
    total_distance += dist
    total_length += length

    dist, length = get_ne_cer(PER + LOC + ORG, P_PER + P_LOC + P_ORG)
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

    per_cnt += len(P_PER)
    loc_cnt += len(P_LOC)
    org_cnt += len(P_ORG)

# --------------- check F1 ---------------




print('------------------TEST RESULTS-------------------')
print('validation set size: {:d}'.format(len(targets)))

print('\n---Character Error Rates---')

printCER()

print('\n---F1 Score Measures---')

printF1()
