from fastai.vision import *
from fastai.metrics import error_rate

# First model using pet images

###########################
####### Get dataset #######
###########################

# Batch size
bs = 64

# help(untar_data)

# print(URLs.PETS)
# URLs.PETS = https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet

path = untar_data(URLs.PETS)

path.ls()

path_anno = path/'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)
fnames[:5]

np.random.seed(2)
pattern = r'/([^/]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(path_img, fnames, pattern, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(7,6))

print(data.classes)
len(data.classes),data.c

###########################
######## Training #########
###########################

# Create the training object
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
# Training model
learn.model
# Trains
learn.fit_one_cycle(4)
# Save result
learn.save('stage-1')

###########################
######## Results ##########
###########################

interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds) == len(losses) == len(idxs)

# Print top losses
interp.plot_top_losses(9, figsize=(15,11))

doc(interp.plot_top_losses)

# Print confusion matrix
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

# Show list of most confused categories
interp.most_confused(min_val=2)

###########################
######## 2nd Round ########
###########################

# Unfreeze to train more
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')

# Prepare chart
learn.lr_find()
# Plot chart
learn.recorder.plot()

learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

###########################
###### Change model #######
###########################

# Bigger images + smaller batch size
data = ImageDataBunch.from_name_re(path_img, fnames, pattern, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)

# Use resnet50
learn = cnn_learner(data, models.resnet50, metrics=error_rate)

# Plot
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(8)

learn.save('stage-1-50')

# Fine-tune
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))

# Use previous model if fine-tune did not help
learn.load('stage-1-50')

###########################
# Interpret results again #
###########################

interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=2)
