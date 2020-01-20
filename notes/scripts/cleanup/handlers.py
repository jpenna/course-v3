import os
import json
import glob
import itertools

def handleImages(handler, path, params = []):
    if (path == '/get_images'):
        pathParam = params.get('path').pop()
        if pathParam is None:
            raise Exception('Missing path param')

        types = ('*.png', '*.jpg')

        files_grabbed = [glob.glob(f'{pathParam}/{ext}') for ext in ['*.png', '*.jpg']]
        files = [f'/{p}' for p in itertools.chain.from_iterable(files_grabbed)];

        return handler.send_json(json.dumps(files))

def handleResource(handler, path):
    full_path = os.getcwd() + path
    if not os.path.exists(full_path):
        handler.send_error(404, 'Not found')
    elif os.path.isfile(full_path):
        handler.handle_file(full_path)
    else:
        handler.send_error(404, 'Unknown object')
