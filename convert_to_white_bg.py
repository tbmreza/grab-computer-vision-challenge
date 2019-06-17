from PIL import Image
import os


removebg_train = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data_removebg/train/'
removebg_test = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data_removebg/test/'
removebg_validation = './dataset/stanford-car-kaggle/stanford-car-dataset-by-classes-folder/car_data_removebg/validation/'

def standard_path(astring):
    if astring.endswith('/'):
        return astring
    return astring+'/'

def convert_to_white_bg():
    for train_test_val in [removebg_train, removebg_test, removebg_validation]:
        for category_folder in os.listdir(train_test_val):
            category_folder = standard_path(category_folder)
            for png_path in os.listdir(train_test_val+category_folder):
                '''
                png_path: 'transparent png' -> 'jpg with white background'
                '''

                png_abspath = train_test_val+category_folder+png_path

                im = Image.open(png_abspath)
                bg = Image.new("RGB", im.size, (255,255,255))
                bg.paste(im,im)

                jpg_path = png_path[:-3]+'jpg'
                jpg_abspath = train_test_val+category_folder+jpg_path
                bg.save(jpg_abspath)
                os.remove(png_abspath)


if __name__ == '__main__':
    convert_to_white_bg()
