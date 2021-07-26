import argparse

from NER.inference import inference


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='NER/data_in', help="Directory containing config.json of data")
    parser.add_argument('--model_dir', default='NER/experiments/base_model_with_crf_val', help="Directory containing config.json of model")

    inference(parser)