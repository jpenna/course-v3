# Models Ref

## Imports

```py
from fastai.vision import *
```

### Files

```py
fnames = get_image_files(path_img)
```

### Labels

The main difference between the handling of image classification datasets is the way labels are stored.
FastAI has several way to retrieve these labels.

```py
#### From CSV

data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)
data.show_batch(rows=3, figsize=(5,5))
data.classes

#### From DATAFRAME

data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
data.classes

#### From REGEX

fn_paths = [path/name for name in df['name']]
pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24, bs=64)
data.classes

#### From FUNCTION

data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
        label_func = lambda x: '3' if '/3/' in str(x) else '7')
data.classes

#### From LISTS

labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)
data.classes
```

#### Batch Size

`bs` param is passed to `ImageDataBunch`.

If you run out of memory, set a smaller `bs` size, so there will be less images being processed together.
It will just take longer to train the model.

### Image visualization

#### Grid

```py
data.show_batch(rows=3, figsize=(7,6))
```

### Training

The model is just a set of coefficients that works for the data set.

> `resnet34` is just a function. It does not store anything, it is just a function.

```py
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4) # Will go through the data set 4x (take care with overfitting)
learn.save('stage-1') # Save based in folders based on dataset, so not to overwrite
```

### Results

```py
# `learn` has the model. This is all this object needs to interpret the results
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(15,11)) # Most confident, but wrong
interp.plot_confusion_matrix(figsize=(12,12), dpi=60) 
interp.most_confused(min_val=2) # Better for big sets. Gets the worst pairs from the confusion matrix
```

Loss functions: tells you how good was your predictions

### More training

In general we will have a pre-trained model and add a few layers on top of it with our data.

We can train the whole model if we want to try new things.

```py
# Tells the learner to train the WHOLE MODEL
learn.unfreeze()
learn.fit_one_cycle(1)
# ... if it gets worse, reload previously saved state
learn.load('stage-1')
```

The issue is we tried to train the whole model in the same speed we were doing before.

```py
# What is the fastest rate we can train the model without ruining it?
learn.lr_find()
learn.recorder.plot() # Plot results
```

This last method will plot a chart of `learning rate` x `loss`.

```py
learn.unfreeze()
# Because the last layers were fine at the default learning rate (1e-3), 
# we can pass a range, so the first layers will take longer but the last will train faster
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4));
```

> *Choosing learning rate rule of thumb*  
> When unfrozen, pass a `max_lr` value with `slice`. 
> Choose the first param as a value 10x smaller than a well before things started getting worse (beginning of the chart), 
> and the second param about 10 times smaller than the first stage.

## Data Bunches

### Sizes

The GPU has to do a bunch of things in a lot of images.
If the images are different sizes or shapes , it can't do this. 
That's why we set a `size` param in `ImageDataBunch`.

`224` is a common size. 

### Sets

2 or 3 datasets with all info :

-  Training data
-  Validation data (set of data the model did not see when training)
-  Test data [optional]

> Validation set prevents overfitting

## Learner

Everything to create a model.
Asks for:

- What is the data?
- What is the model/architecture? (resnet34, resnet50...)

> resnet34: 34 layers  
> resnet50: 50 layers (uses more GPU RAM)

We can also pass a `metrics` param, which tells what the method should print after training the model.
These metrics will be ran against the validation set.

> Resnet is almost always good enough.  
> Better to start with the small one, resnet34, and grow if needed.

>There are specialized architectures if one needs to run mobile.

When it is first run, it downloads the `resnet34` pre-trained weights. 
So we already start with a model that knows how to categorize a thousand of images.

> This is a good practice: take a model that does one thing really well already, 
> and train on top of it to do what you need really well.
> 
> This way one can train in thousands of times faster and with a hundredth of data.

Use the method `fit_one_cycle` to train the model.

> `fit_one_cycle` is better than `fit`, for it runs faster and was a method developed later.

### Layers

Each layer has a semantic complexity. 
The first layer may be horizontal and vertical lines, the second shapes, the third body shapes and so on...
For this reason, it may not be worth training the whole model from scratch, but just using the 
dataset we have to train new layers to recognize what we are looking for in our use case.

Each layer adds a little more detail to identify categories of images.

### Metrics

Train loss: Average distance between the real values (event) and the predicted line (plot)

Valid loss: 

Error rate: % of wrong classification


## What can go wrong?

Most will run right with the defaults, but if there is a problem, it will be most likely related to the `Learning Rate` or `Number of Epochs`.

### Learning Rate

It can be too low or too high.

Too low, besides taking a really long time, may be getting too many looks at each image, so may overfit.

```py
# Default max_lr 3e-3
learn.fit_one_cycle(1, max_lr=0.5) 
# 1. Validation loss goes too high (it is usually below 1)

learn.fit_one_cycle(5, max_lr=1e-5)
# 1. Error rate decreases too slowly
# 2. Training loss > Validation loss: should always be the opposite
#    Learning rate OR Number of epochs is too low

# Plot error rate and validation loss
learn.recorder.plot_losses()
```

### Epochs

Too many or too few epochs.

> Too few epochs and too low learning rate look similar

```py
learn.fit_one_cycle(1)
# 1. Training loss > Validation loss: should always be the opposite
#    Learning rate OR Number of epochs is too low

learn.fit_one_cycle(40)
# Overfitting (error rate improves for a while and starts getting worse)
```

> Training loss < Validation loss IS NOT a sign of overfitting (like some say).
> Actually, the model should be like this.
