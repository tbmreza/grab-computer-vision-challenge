import argparse


WEIGHTS_PATH = '/weights/nets.file'

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Save trained model to WEIGHTS_PATH')
    parser.add_argument('--eval', action='store_true',
                        help='Print the accuracy of training and testing set')
    parser.add_argument('--predict', action='store_true',
                        help='Quick classification')                       
    parser.add_argument('-i','--image', type=str, 
                        help='Specify image path to predict')

    return parser.parse_args()                        


def train_model():
    # Save trained model to WEIGHTS_PATH
    pass

def evaluate_model():
    # Print the accuracy of training and testing set
    pass
    
def predict_image():
    # Quick classification
    print(args.image)
    
if __name__ == '__main__':
    args = set_args()

    if args.train:
        train_model()
    if args.eval:
        evaluate_model()
    if args.predict:
        predict_image()
