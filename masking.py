import soundfile
import yaml
import numpy as np
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

    return (mask_pos[0]-2) / len(parsed_audio), (mask_pos[1]+2) / len(parsed_audio)


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
    with open('TEST/audio_paths.txt') as f:
        audio_paths = [opt['root'] + '/' + line.strip('\n').replace("\\", "/") for line in f.readlines()]

    file_no = 0
    for audio_path, mask_pos in zip(audio_paths, masking_pos):
        signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32') / 32767

        for mp in mask_pos:
            # noise
            duration = int((mp[1] - mp[0]) * signal.shape[0])

            freq_hz = 540.0
            sps = 16000
            esm = np.arange(duration)
            noise = 1 * np.sin(2 * np.pi * esm * freq_hz / sps)

            # configure
            mask_start = int(mp[0] * signal.shape[0])
            front_padding = np.zeros(mask_start)

            mask_end = mask_start + duration
            back_padding = np.zeros(signal.shape[0] - mask_end)

            mask = np.append(front_padding, noise)
            mask = np.append(mask, back_padding)

            # silence original
            signal[mask_start:mask_start + duration] = 0

            masked_signal = signal + mask
            signal = masked_signal
        file_no += 1
        print(audio_path)
        soundfile.write( 'OUTPUTS/audio{:d}.wav'.format(file_no), masked_signal, sps)


if __name__ == "__main__":
    with open('ASR/data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    vocab = KsponSpeechVocabulary(opt['vocab_path'])

    mask_audio(get_masking_pos())

