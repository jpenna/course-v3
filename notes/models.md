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
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
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

### Image visualization

#### Grid

```py
data.show_batch(rows=3, figsize=(7,6))
```

### Training

```py
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
```

### Results

```py
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
```

### More training

```py
learn.unfreeze()
```
