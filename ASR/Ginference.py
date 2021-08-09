import sys
import librosa
import numpy as np
import torch
from numpy import array2string

from ASR.ksponspeech import KsponSpeechVocabulary
from ASR.model.deepspeech import DeepSpeech2
from ASR.utils import check_envirionment, load_model, Timer


def load_audio(audio_path, extension='pcm'):
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    if extension == 'pcm':
        signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')
        return signal / 32767  # normalize audio

    elif extension == 'wav' or extension == 'flac':
        signal, _ = librosa.load(audio_path, sr=16000)
        return signal


def parse_audio(audio_path, opt, audio_extension='pcm'):
    signal = load_audio(audio_path, extension=audio_extension)
    sample_rate = 16000
    frame_length = 20
    frame_shift = 10
    n_fft = int(round(sample_rate * 0.001 * frame_length))
    hop_length = int(round(sample_rate * 0.001 * frame_shift))

    if opt['feature'] == 'melspectrogram':
        feature = librosa.feature.melspectrogram(signal, sample_rate, n_fft=n_fft, n_mels=opt['n_mels'],
                                                 hop_length=hop_length)
        feature = librosa.amplitude_to_db(feature, ref=np.max)

    return torch.FloatTensor(feature).transpose(0, 1)


def inference(opt):
    timer = Timer()
    timer.log('Load Data')
    device = check_envirionment(opt['use_cuda'])
    vocab = KsponSpeechVocabulary(opt['vocab_path'])

    model = DeepSpeech2(
        input_size=opt['n_mels'],
        num_classes=len(vocab),
        rnn_type=opt['rnn_type'],
        num_rnn_layers=opt['num_encoder_layers'],
        rnn_hidden_dim=opt['hidden_dim'],
        dropout_p=opt['dropout_p'],
        bidirectional=opt['use_bidirectional'],
        activation=opt['activation'],
        device=device,
    ).to(device)

    model, optimizer, criterion, scheduler, start_epoch = load_model(opt, model, vocab)


    timer.startlog('Inference Start')

    global filename
    audio_paths = [filename]

    f = open('OUTPUTS/ASR-OUT/transcripts.txt', 'w')
    f2 = open('OUTPUTS/ASR-OUT/parsed_audio.txt', 'w')

    for i, path in enumerate(audio_paths):
        feature = parse_audio(path, opt).to(device)
        input_length = torch.LongTensor([len(feature)])

        model.eval()
        y_hats = model.greedy_search(feature.unsqueeze(0), input_length)

        sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())

        print('sentence no.{:d}'.format(i))
        print(sentence)
        f.write(' '.join(sentence[0].strip("<sos>").split()) + '\n')

        # print(y_hats)
        f2.write(array2string(y_hats.cpu().detach().numpy()[0], threshold=sys.maxsize) + '\n')

    f.close()

    timer.endlog('Inference complete')

