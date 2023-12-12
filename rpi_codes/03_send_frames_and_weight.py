import cv2
import io
import socket
import struct
import time
import pickle
from picamera2 import Picamera2
import RPi.GPIO as GPIO 
from hx711 import HX711
# import zlib


# receiver_ip = '169.254.73.204'
# Slečna Terezka - definice IP a portu
receiver_ip = '169.254.120.100'
receiver_port = 1035
n_values_for_averaging = 2
use_weight = False

if use_weight:
    

    try:
        GPIO.setmode(GPIO.BCM)  # set GPIO pin mode to BCM numbering
        # Create an object hx which represents your real hx711 chip
        # Required input parameters are only 'dout_pin' and 'pd_sck_pin'
        hx = HX711(dout_pin=21, pd_sck_pin=20)
        input('Press Enter to begin reading')

        # measure tare and save the value as offset for current channel
        # and gain selected. That means channel A and gain 128
        err = hx.zero()

        # check if tare was successful
        if err:
            raise ValueError('Tare is unsuccessful.')

        reading = hx.get_raw_data_mean()
        if reading:  # always check if you get correct value or only False
            # now the value is close to 0
            print('Data subtracted by offset but still not converted to units:',
                  reading)
        else:
            print('invalid data', reading)

        # Get conversion of the reading to weight
        input('Put known weight on the scale and then press Enter')
        reading = hx.get_data_mean()
        if reading:
            print('Mean value from HX711 subtracted by offset:', reading)
            known_weight_grams = input('Write how many grams it was and press Enter: ')
            try:
                value = float(known_weight_grams)
                print(value, 'grams')
            except ValueError:
                print('Expected integer or float and I have got:',
                      known_weight_grams)

            # set scale ratio for particular channel and gain which is
            # used to calculate the conversion to units. Required argument is only
            # scale ratio. Without arguments 'channel' and 'gain_A' it sets
            # the ratio for current channel and gain.
            ratio = reading / value  # calculate the ratio for channel A and gain 128
            hx.set_scale_ratio(ratio)  # set ratio for current channel
            print('Ratio is set.')
        else:
            raise ValueError('Cannot calculate mean value. Try debug mode. Variable reading:', reading)
    except:
        pass
else:
    weight = 0.0

# Slečna Terezka - připojení skrze socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((receiver_ip, receiver_port))
connection = client_socket.makefile('wb')
print("Connection established at {}: {}".format(receiver_ip, receiver_port))

cam = Picamera2()
controls = {"ExposureTime": 1000, "AnalogueGain": 1.0, "AwbEnable": False, "ColourGains": (2.25, 2.25), "ScalerCrop": (500, 150, 1000, 1000)}
main = {"size": (1900, 1100)}
capture_config = cam.create_preview_configuration(main = main, controls=controls)
cam.configure(capture_config)
cam.start()
# print(cam.capture_metadata()["ScalerCrop"])
#print(cam.capture_metadata(["ScalerCrop"]))

# img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
# input('Press Enter to begin reading')

# Slečna Terezka - tenhle cyklus sejme obrázek, případně změří váhu, dá je do listu, který zakóduje pomocí pickle a odešle

while True:
    frame = cam.capture_array()
    # frame = frame[300:700, 300:700] 
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    if use_weight:
        weight = hx.get_weight_mean(n_values_for_averaging)
    data_list = [frame, weight]
#    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(data_list, 0)
    size = len(data)


    # print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    # img_counter += 1

cam.release()