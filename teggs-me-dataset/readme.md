# Teggs-me dataset

This is a dataset to train an image classification model. It includes a set of eggs images in the nest. 
The steps followed for build this dataset are specified below.

## Image dataset recollection 

Photos taken with my redmi 6, collected in different days and hours. 

## Resizing images to adequate size

The doc says that the feature extractor can do this... Let's see.

## Dataset objects description

classes: 1 egg, 2 eggs, 3 eggs = 21 / 3 = 7

This is a summary list

| ID | Object identified |
| -- | ----------------- |
| 1 | 2 |
| 2 | 2 |
| 3 | 1 |
| 4 | 1 |
| 5 | 2 | 
| 6 | 2 |
| 7 |  |
| 8 |  |
| 9 |  |
| 10 |  |
| 11 | 1 | 
| 12 | 1 | 
| 13 | 1 |
| 14 | 2 |
| 15 | 2 |
| 16 |  |
| 17 | 1 |
| 18 | 1 |
| 20 | 3 |
| 21 | 2 |
| 22 | 1 |
| 23 | 1 |
| 24 |  |
| 25 | | 

1 = 7
2 = 7
3 = 1 

>blue = 0,0, 255
red = 100, 0, 0
green = 0,100,0

## Image annotation data

There isn't so much information about this. I found some programs that can do it automatically, but I'm not pretty sure about how to use them. So, I decided to do it manually.

I should  create a JSON with each image information, following the COCO Dataset format for **object detection** task.

> * [Source](https://cocodataset.org/#format-data)
> * [About the object detection task](https://cocodataset.org/#detection-2020)

## Dataset definition

