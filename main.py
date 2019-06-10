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


def predict_mask(image_path: str) -> list:
    '''
    Load Mask RCNN model.
    Run detection → r['masks'] → largest_mask
    '''
    largest_mask = ''
    
    return largest_mask



def crop_segment(image: list, mask: list) -> list:
    '''
    Multiply original image with mask to black out pixels other than main object in image
    (as train/eval/test preprocessing)
    Use generator?
    '''
    
    cropped_image = ''
    # cropped_image = image * mask

    return cropped_image

def load_image(image_path: str) -> list:
    '''
    Load image file to array consumable by model
    '''
    pass

class Solution:
    def train_model(self):
        '''
        Save trained model to WEIGHTS_PATH
        '''
        pass

    def evaluate_model(self):
        '''
        Print the accuracy of training and testing set
        '''
        pass
        
    def predict_image(self):
        '''
        Quick classification
        '''
        # Load image as input array and crop

        original_image = load_image(args.image)
        mask = predict_mask(image_path)
        crop_segment(original_image, mask)

        # Load EfficientNet model and weights

        # Output: prediction labels (top-n)
        #         plot cropped_image
    

# TODO:
'''
- that multiplying image with mask returns cropped main object
- predict using out of the box EfficientNet-Pytorch
- train EfficientNet-Pytorch using stanford car dataset
- apply Mask RCNN to tens of sample images
- with/without segmentation results visualization
- fine tune segmentation model for stanford car dataset
'''

if __name__ == '__main__':
    
    args = set_args()
    run = Solution()

    if args.train:
        run.train_model()
    if args.eval:
        run.evaluate_model()
    if args.predict:
        run.predict_image()
