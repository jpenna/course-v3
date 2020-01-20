# Create Dataset

Start importing the lib.

```py
from fastai.vision import *
```

1. Search at Google for the images you want and load as many as possible in the page

```
"canis lupus lupus" -dog -arctos -familiaris -baileyi -occidentalis
```

2. Get all the URLs list running the following in the console

```js
urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```

3. For each category, create a folder and `.csv` file with the URLs.

```py
# Run for each folder
folder = 'grizzly'
file = 'urls_grizzly.csv'

path = Path('data/bears')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)

download_images(path/file, dest, max_pics=200)
# If you have problems download, try with `max_workers=0` to see exceptions:
# download_images(path/file, dest, max_pics=20, max_workers=0)
```

4. Remove all images that cannot be opened

```py
classes = ['teddys','grizzly','black']
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)
```

5. View data [optional]

```py
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

# If you already cleaned the images (topic 8), run the following
# np.random.seed(42)
# data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
#         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

print(data.classes)
# data.show_batch(rows=3, figsize=(7,8))
print(data.classes, data.c, len(data.train_ds), len(data.valid_ds))
# (['black', 'grizzly', 'teddys'], 3, 448, 111)
```

6. Train the model

```py
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
# If the plot is not showing try to give a start and end learning rate
# learn.lr_find(start_lr=1e-5, end_lr=1e-1)
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn.save('stage-2')
```

7. Interpret the results

```py
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

8. Clean up dataset if the wrong results are most likely because of bad selected images

> To run this code, we need to be in GUI environment (jupyter-notebook)
```py
# We can prune our top losses, removing photos that don't belong
from fastai.widgets import *

# First we need to recreate a dataset with all images
db = ImageList.from_folder(path)
              .split_none()
              .label_from_folder()
              .transform(get_transforms(), size=224)
              .databunch()
# If you already cleaned your data using indexes from `from_toplosses`,
# run this cell instead of the one before to proceed with removing duplicates.
# Otherwise all the results of the previous step would be overwritten by
# the new run of `ImageCleaner`.

# db = ImageList.from_csv(path, 'cleaned.csv', folder='.')
#               .split_none()
#               .label_from_df()
#               .transform(get_transforms(), size=224)
#               .databunch()

# Then we create a new learner to use our new databunch with all the images.
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)
learn_cln.load('stage-2')

ds, idxs = DatasetFormatter().from_toplosses(learn_cln)

ImageCleaner(ds, idxs, path)

# Remove duplicates
ds, idxs = DatasetFormatter().from_similars(learn_cln)

# Make sure to recreate the databunch and learn_cln from the cleaned.csv file. 
# Otherwise the file would be overwritten from scratch, 
# losing all the results from cleaning the data from toplosses.
```
 
9. Put the model in production
  
```py
# Export the content of the learner object to a "export.pkl" file
learn.export()
```

> This exported file contains everything we need to deploy our model:
> - Model
> - Weights 
> - And some metadata like the classes or the transforms/normalization used

The machine will use the GPU by default or the CPU if no GPU is available. To force the use of CPU, add the following:

```py
defaults.device = torch.device('cpu')
```

## In production

```py
learn = load_learner(path)

img = open_image(path/'black'/'00000021.jpg')
pred_class,pred_idx,outputs = learn.predict(img)
pred_class # Category black
```
