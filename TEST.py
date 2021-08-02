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


def get_f1_precision(tar, pred):
    if len(pred) == 0:
        return 0, 0
    if len(tar) == 0:
        return 0, len(pred)

    count = 0
    for p in pred:
        if min(map(lambda x: Levenshtein.distance(p, x), tar)) <= 1:
            count += 1

    return count, len(pred)


def get_f1_recall(tar, pred):
    if len(pred) == 0:
        return 0, len(tar)
    if len(tar) == 0:
        return 0, 0

    count = 0
    for t in tar:
        if min(map(lambda x: Levenshtein.distance(t, x), pred)) <= 1:
            count += 1

    return count, len(tar)


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
precision_cnt = 0
precision_len = 0
recall_cnt = 0
recall_len = 0

per_precision_cnt = 0
per_precision_len = 0
per_recall_cnt = 0
per_recall_len = 0

loc_precision_cnt = 0
loc_precision_len = 0
loc_recall_cnt = 0
loc_recall_len = 0

org_precision_cnt = 0
org_precision_len = 0
org_recall_cnt = 0
org_recall_len = 0

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
    p_cnt, p_len = get_f1_precision(PER + LOC + ORG, P_PER + P_LOC + P_ORG)
    r_cnt, r_len = get_f1_recall(PER + LOC + ORG, P_PER + P_LOC + P_ORG)
    precision_cnt += p_cnt
    precision_len += p_len
    recall_cnt += r_cnt
    recall_len += r_len
    print('***')
    print(recall_cnt)
    print(recall_len)

    per_p_cnt, per_p_len = get_f1_precision(PER, P_PER)
    per_r_cnt, per_r_len = get_f1_recall(PER, P_PER)
    per_precision_cnt += per_p_cnt
    per_precision_len += per_p_len
    per_recall_cnt += per_r_cnt
    per_recall_len += per_r_len

    loc_p_cnt, loc_p_len = get_f1_precision(LOC, P_LOC)
    loc_r_cnt, loc_r_len = get_f1_recall(LOC, P_LOC)
    loc_precision_cnt += loc_p_cnt
    loc_precision_len += loc_p_len
    loc_recall_cnt += loc_r_cnt
    loc_recall_len += loc_r_len

    org_p_cnt, org_p_len = get_f1_precision(ORG, P_ORG)
    org_r_cnt, org_r_len = get_f1_recall(ORG, P_ORG)
    org_precision_cnt += org_p_cnt
    org_precision_len += org_p_len
    org_recall_cnt += org_r_cnt
    org_recall_len += org_r_len

print('------------------TEST RESULTS-------------------')
print('validation set size: {:d}'.format(len(targets)))

print('\ntotal CER: {:.3f}'.format(total_distance / total_length))

print('\n-- tags: {:d}'.format(per_cnt + loc_cnt + org_cnt))
print('named-entity CER: {:.3f}'.format(ne_distance / ne_length))

precision = precision_cnt / precision_len
recall = recall_cnt / recall_len
print('\nF1 score: {:.3f}'.format(2 * precision * recall / (precision + recall)))
print('precision: {:.3f}'.format(precision))
print('recall: {:.3f}'.format(recall))

print('\n-- PER tags: {:d}'.format(per_cnt))
per_precision = per_precision_cnt / per_precision_len
per_recall = per_recall_cnt / per_recall_len
if per_cnt > 0:
    print('PER tag CER: {:.3f}'.format(per_distance / per_length))
    print('\nPER F1 score: {:.3f}'.format(2 * per_precision * per_recall / (per_precision + per_recall)))
    print('PER precision: {:.3f}'.format(per_precision))
    print('PER recall: {:.3f}'.format(per_recall))

print('\n-- LOC tags: {:d}'.format(loc_cnt))
loc_precision = loc_precision_cnt / loc_precision_len
loc_recall = loc_recall_cnt / loc_recall_len
if loc_cnt > 0:
    print('LOC tag CER: {:.3f}'.format(loc_distance / loc_length))
    print('\nLOC F1 score: {:.3f}'.format(2 * loc_precision * loc_recall / (loc_precision + loc_recall)))
    print('LOC precision: {:.3f}'.format(loc_precision))
    print('LOC recall: {:.3f}'.format(loc_recall))

print('\n-- ORG tags: {:d}'.format(org_cnt))
org_precision = org_precision_cnt / org_precision_len
org_recall = org_recall_cnt / org_recall_len
if org_cnt > 0:
    print('ORG tag CER: {:.3f}'.format(org_distance / org_length))
    print('\nORG F1 score: {:.3f}'.format(2 * org_precision * org_recall / (org_precision + org_recall)))
    print('ORG precision: {:.3f}'.format(org_precision))
    print('ORG recall: {:.3f}'.format(org_recall))
