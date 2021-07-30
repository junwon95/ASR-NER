import yaml

from ASR.inference import inference


def test():

    with open('ASR/data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt['audio_path'] = 'INPUT/test.pcm'
    opt['use_val_data'] = False
    opt['inference'] = True
    opt['eval'] = False
    inference(opt)

if __name__ == '__main__':
    test()

