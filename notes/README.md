# Fast AI

## fastai

The `fastai` is a framework on top of `Pytorch 1.0`. 
It has a lot of helper methods to make it easier to create AI models.

All the docs can be cloned and ran locally from https://github.com/fastai/fastai_docs.

> Another library that simplify deep learning is Keras.

## Running

### Access GCP

GCP has a fastai docker image, which starts Jupyter automatically on port 8080.

Run the following command to connect to the server and bind local port 8080.

```bash
gcloud compute ssh fastai --zone=us-west2-b -- -L 8080:localhost:8080
```

Open: http://localhost:8080/tree

### Run local

Run the following command from `course-v3` folder (or any parent folder).

```ss
jupyter-notebook
```

## General steps to train a model

To create a model for images:

1. Select images to create a data set
2. Label the images
3. Train data
   1. Use a set for training and a set for validation
4. Verify the results
5. Repeat step 3 to fine tune the model

## Jupyter

### Fastai methods

```
doc(something): show help + link to docs
```
