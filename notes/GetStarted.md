# Fast AI

## fastai

The `fastai` is a framework on top of `Pytorch 1.0`. 
It has a lot of helper methods to make it easier to create AI models.

All the docs can be cloned and ran locally from https://github.com/fastai/fastai_docs.

> Another library that simplify deep learning is Keras.

## Deploy to GCP

```sh
export IMAGE_FAMILY="pytorch-latest-gpu" # or "pytorch-latest-cpu" for non-GPU instances
export ZONE="us-west1-b"
export INSTANCE_NAME="fastai"
export INSTANCE_TYPE="n1-highmem-8" # budget: "n1-highmem-4"

# budget: 'type=nvidia-tesla-k80,count=1'
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-p100,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True"
```

Reference: https://course.fast.ai/start_gcp.html

## Running

### Access GCP

GCP has a fastai docker image, which starts Jupyter automatically on port 8080.

```sh
# Start the instance
gcloud compute instances start fastai

gcloud compute instances start fastai-full
```

Run the following command to connect to the server and bind local port 8080.

```sh
gcloud compute ssh jupyter@fastai --zone=us-west1-b -- -L 8080:localhost:8080
```

Open: http://localhost:8080/tree

When done, stop the instance.

```sh
gcloud compute instances stop fastai
```

#### Update

```sh
# Update repo
cd tutorials/fastai
git pull

# Update fastai lib
sudo /opt/anaconda3/bin/conda install -c fastai fastai
```

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

### Install packages

```py
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install numpy
```

### Dataset

Fast.ai has a list of datasets available, usually from academic papers: [Dataset list](https://course.fast.ai/datasets)
