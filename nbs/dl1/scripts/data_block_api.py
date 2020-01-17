from fastai.vision import *

path = untar_data(URLs.MNIST_TINY)
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=64)

## OR using the data block api
# data = (ImageList.from_folder(path)
#         .split_by_folder()
#         .label_from_folder()
#         .add_test_folder()
#         .transform(tfms, size=64)
#         .databund())

data.show_batch(3, figsize=(6,6), hide_axis=False)
