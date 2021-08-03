import yaml
from ASR.ksponspeech import KsponSpeechVocabulary


def char_to_label(c):
    return vocab.vocab_dict[c]


def get_ne_pos(line):
    ne_label = []
    label_pos = []
    for i, c in enumerate(line):
        if c in '|${':
            s_char = line[i + 1]
            e_idx = i + line[i:].index(']') - 1
            e_char = line[e_idx]

            ne_label.append((char_to_label(s_char), char_to_label(e_char)))
            label_pos.append((line.count(s_char, 0, i + 1), line.count(e_char, 0, e_idx)))

    return ne_label, label_pos


def get_relative_pos(label, position, parsed_audio):
    mask_pos = []
    for c, p in zip(label, position):
        count = 0
        prev = ''
        for i, v in enumerate(parsed_audio):
            if v == c and prev != v:
                if count == p:
                    mask_pos.append(i)
                    break;
                count += 1
            prev = v

    return mask_pos[0] / len(parsed_audio), mask_pos[1] / len(parsed_audio)


def get_masking_pos():
    with open('OUTPUTS/NER-OUT/final_output.txt', 'r', encoding='utf8') as f:
        tagged_transcripts = f.readlines()

    ne_labels = []
    label_positions = []

    for line in tagged_transcripts:
        ne_label, label_pos = get_ne_pos(line)
        ne_labels.append(ne_label)
        label_positions.append(label_pos)

    print(ne_labels)
    print(label_positions)

    with open('OUTPUTS/ASR-OUT/parsed_audio.txt', 'r') as f:
        parsed_audios = [line[1:].split() for line in f.read().split(']\n')]

    masking_pos = []

    for labels, positions, parsed_audio in zip(ne_labels, label_positions, parsed_audios):
        mask_pos = []
        for label, position in zip(labels, positions):
            mask_pos.append(get_relative_pos(label, position, parsed_audio))
        masking_pos.append(mask_pos)

    return masking_pos


def mask_audio(masking_pos):



if __name__ == "__main__":
    with open('ASR/data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    vocab = KsponSpeechVocabulary(opt['vocab_path'])

    get_masking_pos()
