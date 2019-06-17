import argparse
import csv
import os

from efficientnet import EfficientNetB3 as Net
from keras import Sequential, layers, optimizers
from numpy import argmax
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

from IPython.display import Image
import matplotlib.pyplot as plt
# IPython magic command: %matplotlib inline
# get_ipython().run_line_magic('matplotlib', 'inline')

BATCH_SIZE = 32
WIDTH = 150
HEIGHT = 150
EPOCHS = 20
NUM_TRAIN = 8144
NUM_TEST = 7453
DROPOUT_RATE = 0.2
INPUT_SHAPE = (HEIGHT, WIDTH, 3)
NFROZEN_LAYERS = 300

names_list = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/names.csv'

def generate_dataset(remove_background: bool):
    global train_dir, validation_dir, test_dir
    train_dir = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data/train'
    validation_dir = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data/validation'
    test_dir = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data/test'
    if remove_background:
        train_dir = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data_removebg/train'
        validation_dir = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data_removebg/validation'
        test_dir = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data_removebg/test'

    train_datagen = ImageDataGenerator(rescale=1./255)
    global train_generator
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

    validation_datagen = ImageDataGenerator(rescale=1./255)
    global validation_generator
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')


def compile_model(h5_file: str, pretrained=False):
    conv_base = Net(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    '''
    Set and print number of frozen layers of base efficientnet model.
    '''
    plot_model(conv_base, to_file='./output/conv_base.png', show_shapes=True)
    Image(filename='./output/conv_base.png')

    print(f'\nconv_base.layers length: {len(conv_base.layers)}')

    for layer in conv_base.layers[:NFROZEN_LAYERS]:
        layer.trainable = False

    trainable_layers = 0
    for layer in conv_base.layers:
        if layer.trainable:
            trainable_layers += 1

    print(f'Left last {trainable_layers} layers of conv_base trainable.')

    '''
    After these base layers are added to new Sequential(),
    they will be interpreted as a unit.
    '''
    global model
    model = Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D())
    if DROPOUT_RATE: model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.Dense(196, activation="softmax"))

    print(f'model.layers length: {len(model.layers)}')

    if pretrained:
        model.load_weights(h5_file)
    else:
        '''
        Compile and train model.
        '''
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
                    metrics=['accuracy'])

        plot_model(model, to_file='./output/model.png', show_shapes=True)
        Image(filename='./output/model.png')

        global history
        history = model.fit_generator(
            train_generator,
            steps_per_epoch= NUM_TRAIN //BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps= NUM_TEST //BATCH_SIZE,
            verbose=1,
            use_multiprocessing=True,
            workers=4)

        os.makedirs("./models", exist_ok=True)
        model.save(h5_file)


def plot_epochs():
    ACC = history.history['acc']
    VAL_ACC = history.history['val_acc']
    LOSS = history.history['loss']
    VAL_LOSS = history.history['val_loss']
    EPOCHS_X = range(len(ACC))

    plt.plot(EPOCHS_X, ACC, 'bo', label='Training acc')
    plt.plot(EPOCHS_X, VAL_ACC, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(EPOCHS_X, LOSS, 'bo', label='Training loss')
    plt.plot(EPOCHS_X, VAL_LOSS, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def test_accuracy():
    def standard_path(astring) -> str:
        if astring.endswith('/'):
            return astring

        return astring+'/'


    def csv_to_dicts(filename) -> dict:
        class_dict = {}
        with open(filename, mode='r') as infile:
            reader = csv.reader(infile)
            class_index = 0
            for row in reader:
                class_dict[class_index] = row
                class_index += 1

        return class_dict


    def predict_image(image_path) -> list:
        img = image.load_img(image_path, target_size=(HEIGHT, WIDTH))
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        x /= 255.
        top1_index = argmax(model.predict([x])[0])
        class_dict = csv_to_dicts(names_list)
        verdict = class_dict[top1_index]
        
        return verdict[0]

    '''
    Run predict_image() folder by folder.
    '''
    print(f'Reading test_dir={test_dir}')
    correct = 0
    categories = os.listdir(test_dir)
    total = len(categories)
    assert total == 196

    for folder in categories:
        folder_path = standard_path(test_dir)+standard_path(folder)
        for image_file in os.listdir(folder_path):
            image_path = folder_path+image_file
            verdict = predict_image(image_path)
            
            if folder.lower() == verdict.lower():
                correct += 1
            else:
                print(f'Wrong: folder = {folder}, verdict = {verdict}')

    print(f'Test accuracy: {correct/total*100}% ({correct}/588)')


class Solution:
    def __init__(self, remove_background: bool):
        self.remove_background = remove_background
        
        if self.remove_background:
            self.h5_file = './models/efficientnet_stanford_car_removebg.h5'
        else:
            self.h5_file = './models/efficientnet_stanford_car.h5'

    def train_model(self):
        '''
        Plot model training and validation loss and accuracy
        '''
        generate_dataset(self.remove_background)
        compile_model(self.h5_file)


    def predict_cars(self):
        '''
        Load model from WEIGHTS_FILE then output test data accuracy
        '''
        compile_model(self.h5_file, pretrained=True)
        test_accuracy()

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Plot model training and validation loss and accuracy')
    parser.add_argument('--predict', action='store_true',
                        help='Load model from WEIGHTS_FILE then output test data accuracy')                       
    parser.add_argument('--removebg', action='store_true',
                        help='Set image background removal preprocessing')
    
    return parser.parse_args()                        

if __name__ == '__main__':
    '''
    python3 main.py --removebg --train --predict
    '''
    args = set_args()
    
    remove_background = False
    if args.removebg:
        remove_background = True
    
    run = Solution(remove_background)

    if args.train:
        run.train_model()
        plot_epochs()
    if args.predict:
        assert os.path.exists(run.h5_file)
        run.predict_cars()
