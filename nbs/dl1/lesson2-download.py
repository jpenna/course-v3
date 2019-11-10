from fastai.vision import *

details = [
    { 'folder': 'trucks', 'label': 'url_trucks.csv' },
    { 'folder': 'toys', 'label': 'url_toys.csv' },
    { 'folder': 'pickups', 'label': 'url_pickups.csv' },
    { 'folder': 'cars', 'label': 'url_cars.csv' },
]


path = Path('data/1_trucks')

for detail in details:
    dest = path/detail['folder']
    dest.mkdir(parents=True, exist_ok=True)
    download_images(path/detail['label'], dest, max_workers=0, max_pics=20)
    verify_images(dest, delete=True, max_size=500)

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


