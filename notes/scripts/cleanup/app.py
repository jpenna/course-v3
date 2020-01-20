from http.server import BaseHTTPRequestHandler, HTTPServer
import handlers
import os
import re
from urllib.parse import parse_qs
import functools

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        (path, query) = self.getPathAndQuery()
        try:
            if path == '/notes/scripts/cleanup/get_images':
                handlers.handleGetImages(self, path, query)
                return
            handlers.handleResource(self, path)
        except Exception as msg:
            self.send_error(500, str(msg))

    def do_POST(self):
        (path, query) = self.getPathAndQuery()
        contentLength = int(self.headers.get('Content-Length'))
        # Replace read to receive content length when called by json.load
        self.rfile.read = functools.partial(self.rfile.read, contentLength)
        try:
            if path == '/notes/scripts/cleanup/images':
                handlers.moveImage(self, self.rfile)
            else:
                self.send_error(404, 'Route not found')
        except Exception as msg:
            self.send_error(500, str(msg))

    def do_DELETE(self):
        (path, query) = self.getPathAndQuery()
        try:
            if path == '/notes/scripts/cleanup/images':
                handlers.deleteImage(self, query)
            else:
                self.send_error(404, 'Route not found')
        except Exception as msg:
            self.send_error(500, str(msg))

    def getPathAndQuery(self):
        path = re.split(r'/?\?', self.path);
        query = parse_qs(path[1]) if len(path) > 1 else None;
        return (path[0], query)

    def handle_file(self, full_path):
        try:
            with open(full_path, 'rb') as reader:
                content = reader.read()
            self.send_file(content)
        except IOError as msg:
            msg = "'{0}' cannot be read: {1}".format(self.path, str(msg))
            self.send_error(500, msg)

    def send_file(self, content, status=200):
        self.send_response(status)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, content, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content.encode())


if __name__ == "__main__":
    port = int(os.getenv('PORT', 7890))
    serverAddress = ('', port)
    try:
        server = HTTPServer(serverAddress, RequestHandler)
        print(f'Server listen on http://localhost:{port}')
        server.serve_forever()
    except Exception as err:
        print(err)
