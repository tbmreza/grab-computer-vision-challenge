# grab-computer-vision-challenge

My attempt on [aiforsea](https://www.aiforsea.com/challenges)»[computer-vision](https://www.aiforsea.com/computer-vision) challenge.

<img src="https://raw.githubusercontent.com/tbmreza/grab-computer-vision-challenge/master/readme/Grab_EDM_Computer_Vision.webp?raw=true" alt="Computer vision challenge logo">

## Approach

First thoughts:

- Use state of the art method for image classification with necessary modification.
- Idea for solution originality: does segmentation improve classification accuracy?

This solution description:

- Downloaded better organized dataset from [this kaggle post](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder).
- More engineering than pure research.
- Tries applying segmentation prior to classification because ['segmenting an image does improve object categorization
accuracy.'](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.310.6542)
- Segmentation: [Xception and MobileNet](https://github.com/susheelsk/image-background-removal) (A small portion of the training set are segmented using Xception. The rest are segmented using MobileNet that runs less than a second instead of 4 seconds each image), Classification: [EfficientNet-B3](https://github.com/qubvel/efficientnet).
- The segmentation model is not evaluated.

## Requirements

- keras==2.2.0 (version 2.2.4 may raise _ImportError: cannot import name 'GeneratorEnqueuer'_)
- tensorflow==1.10.0
- pillow==4.0.0

## Train, validate, test model

```sh
python3 main.py --removebg --train --predict
```
- `--train` saves trained model to `models/{h5_filename}`
- `--predict` loads model weights and predict images in test set folder.
- `--removebg` sets true segmentation preprocessing for train/validation/test.

Notebook equivalent to `python3 main.py --train --predict` or `python3 main.py --removebg --train --predict` is also provided.

## Results

Due to hardware limitations, I had to narrow down the scope of this solution from both achieving accuracy as high as possible and comparing the model performance with/without segmentation preprocessing to only the latter. Practically, I set `BATCH_SIZE` and `learning rate` to suboptimal numbers to achieve quicker computation time.


<table>

<tr>
<th>&nbsp;</th>
<th>Without preprocessing</th>
<th>With preprocessing</th>
</tr>

<!-- Line 1: Train graph -->
<tr>
<td><em>Train graph</em></td>
<td><img src="https://raw.githubusercontent.com/tbmreza/grab-computer-vision-challenge/master/readme/train_graph.jpg?raw=true" alt="train_graph"></td>
<td><img src="https://raw.githubusercontent.com/tbmreza/grab-computer-vision-challenge/master/readme/train_graph_preprocessing.jpg?raw=true" alt="train_graph_preprocessing"></td>
</tr>

<!-- Line 2: Validation graph -->
<tr>
<td><em>Validation graph</em></td>
<td><img src="https://raw.githubusercontent.com/tbmreza/grab-computer-vision-challenge/master/readme/validation_graph.jpg?raw=true" alt="train_graph"></td>
<td><img src="https://raw.githubusercontent.com/tbmreza/grab-computer-vision-challenge/master/readme/validation_graph_preprocessing.jpg?raw=true" alt="validation_graph_preprocessing"></td>
</tr>

<!-- Line 3: Test accuracy -->
<tr>
<td><em>Test accuracy*</em></td>
<td>80%</td>
<td>82%</td>
</tr>

</table>

*most recent run on my local machine.

How I split the dataset:
- Initially the dataset from [this kaggle post](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder) is roughly 50-50 split with 8,144 train and 8,041 test images.
- Left the **train** set folder as is.
- Renamed the other folder to **validation**.
- Cut 3 images from each classes to new folder named **test**.
- **train**: 8,144 **validation**: 7,453 **test**: 3*196=588

## Acknowledgement

- https://github.com/Tony607/efficientnet_keras_transfer_learning
- https://github.com/qubvel/efficientnet
- https://github.com/susheelsk/image-background-removal