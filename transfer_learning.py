import argparse
import os

from . import transfer_learning as tl

WEIGHTS_FILE = './models/efficientnet_stanford_car.h5'

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Plot model training and validation loss and accuracy')
    parser.add_argument('--predict', action='store_true',
                        help='Load model from WEIGHTS_FILE then output test data accuracy')                       
    parser.add_argument('--removebg', action='store_true',
                        help='Set image background removal preprocessing')
    
    return parser.parse_args()                        


class Solution:
    def train_model(self, removebg: bool):
        '''
        Plot model training and validation loss and accuracy
        '''

        tl.generate_dataset(self.removebg)
        tl.compile_model()
        tl.plot_epochs()


    def predict_cars(self, removebg: bool):
        '''
        Load model from WEIGHTS_FILE then output test data accuracy
        '''

        tl.load_trained_model(self.removebg)
        tl.test_accuracy()


if __name__ == '__main__':
    
    args = set_args()
    remove_background = False
    
    run = Solution()

    if args.removebg:
        remove_background = True
        WEIGHTS_FILE = './models/efficientnet_stanford_car_removebg.h5'
    if args.train:
        run.train_model(remove_background)
    if args.predict:
        assert os.path.exists(WEIGHTS_FILE)
        run.predict_cars(remove_background)


