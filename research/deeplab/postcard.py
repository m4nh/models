import binascii
import socket
import struct
import sys
import signal
import os
import numpy as np
import cv2
import threading
import time
from PIL import Image
import io


class ImageManipulator(object):

    def byteArrayToImage(buffer):
        pilimage = Image.open(io.BytesIO(buffer))
        return np.array(pilimage)

    def imageToByteArray(image, format='png'):
        buffer = io.BytesIO()
        pilimage = Image.fromarray(image)
        pilimage.save(buffer, format=format)
        buffer = buffer.getvalue()
        return buffer


class PostcardImage(object):
    IMAGE_FORMAT_JPG = 'jpeg'
    IMAGE_FORMAT_PNG = 'png'

    def __init__(self, raw_image, format):
        self.raw_image = raw_image
        self.format = format
        self.bytes = ImageManipulator.imageToByteArray(self.raw_image, self.format)


class Header(object):
    """ Header Object. It represents the base exchanged message """
    HEADER_COMMAND_LEN = 32
    HEADER_SIZE = 52
    HEADER_FORMAT = fmt = 'BBBBiiii'+('B'*32)

    def __init__(self, raw_data=None):
        if raw_data is not None:
            self.received_data = Header.unpacker().unpack(raw_data)
            self.crc = self.received_data[0:4]
            self.width = self.received_data[4]
            self.height = self.received_data[5]
            self.depth = self.received_data[6]
            self.byte_per_element = self.received_data[7]
            self.command = str(
                bytes(self.received_data[8:]), 'utf-8').rstrip('\0')
            self.command = self.command.strip()
        else:
            self.received_data = None
            self.crc = (0, 0, 0, 0)
            self.width = 0
            self.height = 0
            self.depth = 0
            self.byte_per_element = 0
            self.command = ""

    def pack(self):
        """ Pack Header in a Byte Array """
        command_ext = self.command.ljust(Header.HEADER_COMMAND_LEN)
        command_ext = bytes(command_ext, 'utf-8')
        print(command_ext, len(command_ext), command_ext)
        return struct.pack(Header.HEADER_FORMAT,
                           *self.crc,
                           self.width,
                           self.height,
                           self.depth,
                           self.byte_per_element,
                           *command_ext
                           )

    @staticmethod
    def unpacker():
        """ Builds the unpacker python object """
        return struct.Struct(Header.HEADER_FORMAT)


class PostcardServer(object):
    ACTIVE_THREADS = []

    def __init__(self, connection, client_address, data_callback=None, auto_start=True):
        self.connection = connection
        self.client_address = client_address
        self.thread = threading.Thread(target=self.run)
        PostcardServer.ACTIVE_THREADS.append(self.thread)
        if auto_start:
            self.thread.start()
        self.data_callback = data_callback

    def start(self):
        self.thread.start()

    def run(self):
        while True:
            # Receive Header
            print("Wairting for new Header")
            data = self.connection.recv(Header.HEADER_SIZE)
            header = Header(data)
            print("REC", header.crc, header.width, header.height,
                  header.depth, header.byte_per_element, header.command)

            payload_size = header.width*header.height*header.depth*header.byte_per_element

            received_image = None
            if payload_size > 0:

                # Receive Payload
                print("Wairting for ", payload_size, "bytes")
                received_size = 0
                received_data = b''
                while len(received_data) < payload_size:
                    chunk = self.connection.recv(payload_size-received_size)
                    if not chunk:
                        break
                    received_data += chunk

                print("Payload received", len(received_data))

                # Building Image

                if header.height == 1:
                    received_image = ImageManipulator.byteArrayToImage(received_data)
                else:
                    received_image = np.frombuffer(received_data, dtype=np.uint8)
                    received_image = received_image.reshape((
                        header.height,
                        header.width,
                        header.depth
                    ))

            command = ""
            response_image = None
            if self.data_callback is not None:
                command, response_image = self.data_callback(
                    header,
                    received_image
                )
            self.sendResponse(command, response_image)

    def sendResponse(self, command, image=None):
        response_header = Header()

        response_header.command = command
        if image is not None:
            if isinstance(image, PostcardImage):
                response_header.width = len(image.bytes)
                response_header.height = 1
                response_header.depth = 1
                response_header.byte_per_element = 1
            else:
                response_header.height = image.shape[0]
                response_header.width = image.shape[1]
                response_header.depth = image.shape[2] if len(image.shape) > 2 else 1
                response_header.byte_per_element = 1
        else:
            response_header.height = 0
            response_header.width = 0
            response_header.depth = 0
            response_header.byte_per_element = 0

        self.connection.send(response_header.pack())
        print("Response Header Sent!")
        if image is not None:
            if isinstance(image, PostcardImage):
                self.connection.send(image.bytes)
            else:
                self.connection.send(image.tobytes())
        print("Sent!")

    @staticmethod
    def newServer(socket, data_callback):
        connection, address = socket.accept()
        return PostcardServer(connection, address, data_callback=data_callback)

    @staticmethod
    def AcceptingSocket(host='0.0.0.0', port=8000):
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_address = (host, port)
        sock.bind(server_address)
        sock.listen(1)
        return sock


class PostcardClient(object):

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((host, port))
        self.connection = self._socket
        print("CONNECTION", self.connection)

    def sendImage(self, command, image):

        print("Sending image")
        response_header = Header()

        response_header.command = command
        if image is not None:
            if isinstance(image, PostcardImage):
                response_header.width = len(image.bytes)
                response_header.height = 1
                response_header.depth = 1
                response_header.byte_per_element = 1
            else:
                response_header.height = image.shape[0]
                response_header.width = image.shape[1]
                response_header.depth = image.shape[2] if len(image.shape) > 2 else 1
                response_header.byte_per_element = 1
        else:
            response_header.height = 0
            response_header.width = 0
            response_header.depth = 0
            response_header.byte_per_element = 0

        self.connection.send(response_header.pack())
        if image is not None:
            if isinstance(image, PostcardImage):
                self.connection.send(image.bytes)
            else:
                self.connection.send(image.tobytes())

        # Receive Header
        print("Wairting for new Header")
        data = self.connection.recv(Header.HEADER_SIZE)
        header = Header(data)
        print("REC", header.crc, header.width, header.height,
              header.depth, header.byte_per_element, header.command)

        payload_size = header.width*header.height*header.depth*header.byte_per_element

        received_image = None
        if payload_size > 0:

            # Receive Payload
            print("Wairting for ", payload_size, "bytes")
            received_size = 0
            received_data = b''
            while len(received_data) < payload_size:
                chunk = self.connection.recv(payload_size-received_size)
                if not chunk:
                    break
                received_data += chunk

            print("Payload received", len(received_data))

            # Building Image

            if header.height == 1:
                received_image = ImageManipulator.byteArrayToImage(received_data)
            else:
                received_image = np.frombuffer(received_data, dtype=np.uint8)
                received_image = received_image.reshape((
                    header.height,
                    header.width,
                    header.depth
                ))

        command = ""
        response_image = None
        return header, received_image


def ConnectionWorker(connection):
    while True:
        try:

            # Receive Header
            print("Wairting for new Header")
            data = connection.recv(Header.HEADER_SIZE)
            header = Header(data)
            print("REC", header.crc, header.width, header.height,
                  header.depth, header.byte_per_element, header.command)

            payload_size = header.width*header.height*header.depth*header.byte_per_element

            # Receive Payload
            print("Wairting for ", payload_size, "bytes")
            received_size = 0
            received_data = b''
            while len(received_data) < payload_size:
                chunk = connection.recv(payload_size-received_size)
                if not chunk:
                    break
                received_data += chunk

            print("Payload received", len(received_data))

            # Building Image
            mat = np.frombuffer(received_data, dtype=np.uint8)
            mat = mat.reshape((
                header.height,
                header.width,
                header.depth
            ))

            # Computing response
            print("Computing response...")
            edges = cv2.Canny(mat, 100, 200)

            # Prepare response
            print("Response header")
            response_header = Header()

            response_header.command = "result_image"
            response_header.height = edges.shape[0]
            response_header.width = edges.shape[1]
            response_header.depth = edges.shape[2] if len(
                edges.shape) > 2 else 1
            response_header.byte_per_element = 1

            connection.send(response_header.pack())
            connection.send(edges.tobytes())

        except Exception as e:
            print(e)
            connection.close()
            break
    return