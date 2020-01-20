import os
import json

def handleImages(handler):
    if (handler.path == '/get_images'):
        return handler.send_json(json.dumps({ 'my': 1 }))

def handleResource(handler):
    full_path = os.getcwd() + handler.path
    if not os.path.exists(full_path):
        handler.send_error(404, 'Not found')
    elif os.path.isfile(full_path):
        handler.handle_file(full_path)
    else:
        handler.send_error(404, 'Unknown object')
