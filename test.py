import yaml

from ASR.inference import inference
from NER.Interactive_shell_NER import NER


def test():

    with open('ASR/data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt['audio_path'] = 'INPUT/test.wav'
    opt['use_val_data'] = False
    opt['inference'] = True
    opt['eval'] = True
    inference(opt)




if __name__ == '__main__':
    test()
    NER()
