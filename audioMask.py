import soundfile
import numpy as np

audio_path = r'INPUT/test.pcm'

mask_point = [(0.1, 0.2), (0.4, 0.7)]

signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32') / 32767

for mp in mask_point:

    # noise
    duration = int((mp[1]-mp[0])*signal.shape[0])

    freq_hz = 540.0
    sps = 16000
    esm = np.arange(duration)
    noise = 1 * np.sin(2 * np.pi * esm * freq_hz / sps)

    # configure
    mask_start = int(mp[0]*signal.shape[0])
    front_padding = np.zeros(mask_start)

    mask_end = mask_start + duration
    back_padding = np.zeros(signal.shape[0] - mask_end)

    mask = np.append(front_padding, noise)
    mask = np.append(mask, back_padding)

    # silence original
    signal[mask_start:mask_start+duration] = 0

    masked_signal = signal + mask
    signal = masked_signal

soundfile.write('OUTPUTS/filename.wav', masked_signal, sps)