from fastai.vision import *
from fastai.metrics import error_rate

# Using other data formats

###########################
####### Get dataset #######
###########################

#### From FOLDER

# /course-v3/nbs/dl1/data/mnist_sample
path = untar_data(URLs.MNIST_SAMPLE)

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)

# Show image grid
data.show_batch(rows=3, figsize=(5,5))

# Learn
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)

# Read labels
df = pd.read_csv(path/'labels.csv')
# Print labels
df.head()

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
