import os
import json
import glob
import re
import itertools
import random

def handleGetImages(handler, path, query = []):
    pathParam = query.get('path').pop()
    if pathParam is None:
        raise Exception('Missing path param')

    types = ('*.png', '*.jpg')

    files_grabbed = [glob.glob(f'{pathParam}/{ext}') for ext in ['*.png', '*.jpg']]
    files = [f'/{p}' for p in itertools.chain.from_iterable(files_grabbed)];

    handler.send_json(json.dumps(files))

def handleResource(handler, path):
    full_path = os.getcwd() + path
    if not os.path.exists(full_path):
        handler.send_error(404, 'Not found')
    elif os.path.isfile(full_path):
        handler.handle_file(full_path)
    else:
        handler.send_error(404, 'Unknown object')

def deleteImage(handler, query):
    full_path = os.getcwd() + query['path'][0]
    if not os.path.exists(full_path):
        handler.send_error(404, 'Not found')
    elif os.path.isfile(full_path):
        os.remove(full_path)
        handler.send_json(json.dumps({ 'success': True }))
    else:
        handler.send_error(404, 'Not found')


def moveImage(handler, postBodyObject):
    info = json.load(postBodyObject)

    if not info['path'] or not info['newCategory']:
        raise Exception('Missing query')

    prevPath = os.getcwd() + info['path']
    newCategory = info['newCategory']
    randomName = random.randrange(999999999)
    newFileName = re.sub(r'.+\.', f'{randomName}.', info['path'])
    newDir = os.getcwd() + re.sub(r'\/\w+\/.[\w.]+$', f'/{newCategory}', info['path'])
    newPath = f'{newDir}/{newFileName}'

    print(newDir)
    print(newPath)

    if not os.path.exists(prevPath):
        handler.send_error(404, 'Not found')
    if not os.path.exists(newDir):
        handler.send_error(404, 'New path not found')
    elif os.path.isfile(prevPath):
        os.rename(prevPath, newPath)
        handler.send_json(json.dumps({ 'success': True }))
    else:
        handler.send_error(404, 'Not found')
