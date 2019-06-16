# grab-computer-vision-challenge

My attempt on [aiforsea](https://www.aiforsea.com/challenges)[»computer-vision](https://www.aiforsea.com/computer-vision) challenge.

<img src="https://raw.githubusercontent.com/tbmreza/grab-computer-vision-challenge/master/readme/Grab_EDM_Computer_Vision.webp?raw=true" alt="Computer vision challenge logo">

## Requirements

- keras==2.2.0
- tensorflow==1.10.0
- pillow==4.0.0

## Approach

First thoughts:

- Use state of the art method for image classification with necessary modification.
- Idea for solution originality: does segmentation improve classification accuracy?

This solution description:

- Downloaded better organized dataset from [kaggle](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder).
- More engineering than pure research.
- Tries applying segmentation prior to classification because ['segmenting an image does improve object categorization
accuracy.'](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.310.6542)
- Segmentation: [Xception and MobileNet](https://github.com/susheelsk/image-background-removal), Classification: [EfficientNet](https://github.com/qubvel/efficientnet).
- The segmentation model is not evaluated.


## Train, validate, test model

```sh
python3 main.py --removebg --train --predict
```
- `--train` saves trained model to `models/{h5_filename}`
- `--predict` loads model weights and predict images in test set folder.
- `--removebg` sets true segmentation preprocessing for train/validation/test.

Notebook equivalent to `python3 main.py --train --predict` or `python3 main.py --removebg --train --predict` is also provided.

## Results

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

## Acknowledgement

- https://github.com/Tony607/efficientnet_keras_transfer_learning
- https://github.com/qubvel/efficientnet
- https://github.com/susheelsk/image-background-removal