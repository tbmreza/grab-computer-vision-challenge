# grab-computer-vision-challenge
Given a dataset of distinct car images, can you automatically recognize the car model and make?

## Approach

First thought:

- Use state of the art method for image classification with necessary modification.
- Does segmentation improve classification accuracy?

This solution description:

- More engineering than pure research.
- Applies segmentation prior to classification because ['segmenting an image does improve object categorization
accuracy.'](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.310.6542)
- Segmentation: [Mask RCNN](https://github.com/matterport/Mask_RCNN), Classification: [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch).
- The masking model is not evaluated.

### Segment image (for train/eval/test preprocessing)
- Load Mask RCNN model.
- Run detection → r['masks'] → largest_mask
- def segment(image, largest_mask): ... return 'object with black BG.jpg'

## Train model

Saves weights to WEIGHTS_PATH

```sh
python main.py --train
```

## Evaluate model

```sh
python3 main.py --eval
```