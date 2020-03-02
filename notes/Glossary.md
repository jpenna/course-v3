# Glossary

## Accuracy

`correctly predicted values` / `total number of values`

## Epoch

One complete run through all of our data points. 

Every epoch is a pass through the images. The more epochs, the bigger the chance of overfitting. So generally we don't want to do too many epochs.

## Learning Rate

The value we multiply the gradient ("slope") to decide how much to update the weight.

## Loss function

Tells how far or close we are to the correct answer.

## Minibatch

A set of random data points used to update the weight.

## Model / Architecture

The mathematical function we are fitting the parameters.

## Parameters (coefficients, weights)

The values we are updating.

## Progressive resizing

[Video](https://youtu.be/MpZxV6DVsmM?t=4251)

Start with images resized to a small size (64x64), then progressively increase the size of the images (128x128, 256x256...) and train new layers for each size. The final result is improved by doing this. 

## Regression model

Any kind of model where your output is some continuous number or set of numbers.

## Segmentation

[Video](https://youtu.be/MpZxV6DVsmM?t=3400)

Do a classification of every pixel in every single image. A dataset with pixels labeled would be required to train a model for this, but we can find models and go from them. 

CAMVID is a dataset we can use for video for example ([lesson3-camvid](../nbs/dl1/lesson3-camvid.ipynb))

## SGD (Stochastic gradient descent)

Gradient descent (approximate the local minimum of a differentiable function) using minibatches.

## Tensor

In AI, tensor is a structured array. If it is 4x3, all the rows and columns have expected data.
