
from models.CNN import CNN
from  Params import Params
import argparse

if __name__ == '__main__':
    params = Params()
    parser = argparse.ArgumentParser(description='Set configuration file.')
    parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
    args = parser.parse_args()
    params.parse_config(args.config)

    cnn = CNN(params)
    model = cnn.get_model(opt=params)
    model.save('try.h5')

