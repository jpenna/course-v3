from http.server import BaseHTTPRequestHandler, HTTPServer
from handlers import handleImages, handleResource
import os

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if (handleImages(self)):
                return
            handleResource(self)
        except Exception as msg:
            self.send_error(500, str(msg))

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
    port = int(os.getenv('PORT', 8080))
    serverAddress = ('', port)
    try:
        server = HTTPServer(serverAddress, RequestHandler)
        print(f'Server listen on http://localhost:{port}')
        server.serve_forever()
    except Exception as err:
        print(err)
