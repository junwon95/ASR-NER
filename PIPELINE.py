import yaml

from ASR.inference import inference
from NER.Interactive_shell_NER import NER


def ASR():

    with open('ASR/data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt['use_val_data'] = False
    opt['inference'] = True
    opt['eval'] = True
    inference(opt)




if __name__ == '__main__':
    ASR()
    NER()
