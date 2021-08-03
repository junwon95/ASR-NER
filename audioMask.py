import soundfile
import numpy as np

audio_path = r'INPUT/test.pcm'

signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')/ 32767


# noise
duration = 30000

freq_hz = 540.0
sps = 16000
esm = np.arange(duration)
noise = 1 * np.sin(2 * np.pi * esm * freq_hz / sps)

# configure
mask_point = 20368
front_padding = np.zeros(20368)

end_point = 50368
back_padding = np.zeros(signal.shape[0] - end_point)

mask = np.append(front_padding, noise)
mask = np.append(mask, back_padding)

masked_signal = signal + mask
soundfile.write('OUTPUTS/filename.wav', masked_signal, sps)