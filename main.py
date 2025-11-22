import numpy as np
import os
import cv2

import math

def class_color(byte):
    color = tuple()
    if byte == 0:
        color = (128, 128, 128) 
    elif byte == 255:
        color = (255, 255, 0)
    elif 1 <= byte <= 31 or byte == 127:
        color = (0, 255, 0)
    elif 32 <= byte <= 126:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    return color

def color_to_image(color_list):
    img = np.zeros((10, 110, 3), dtype=np.uint8)

    for i, (r, g, b) in enumerate(color_list):
        start = i * 10
        end = start + 10
        img[:, start:end] = (b, g, r)
    return img
    
def fuzzy_inference_system(current_color, left_side, right_side):    
    count = 0
    print(current_color, left_side, right_side)

    for color_byte in left_side:
        if np.array_equal(current_color, color_byte):
            count += 1
        pass
    left_similarity = np.array([ int(np.array_equal(current_color, color_byte)) for color_byte in left_side ]).sum() / len(left_side)
    right_similarity = np.array([ int(np.array_equal(current_color, color_byte)) for color_byte in right_side ]).sum() / len(right_side)
    
    
    print(left_similarity, right_similarity)
    
    return 0.0;
    
# Signature Agnostic Binary Visualizer
def SABV(color_list):
    N = 5    
    new_list = np.zeros((10, 3), dtype=np.uint8)

    color_list_length = len(color_list)

    
    i = 5
    left_side = color_list[max(0, i - N):i]
    right_side = color_list[i + 1:min(color_list_length, i + N + 1)]
    
    fuzzy_inference_system(color_list[i], left_side, right_side)
        
    return new_list;

                

if __name__ == "__main__":
    # test path
    file_path = os.getcwd() + "/PE-files/546.exe"    
    with open(file_path, 'rb') as file:
        byte_array = np.frombuffer(file.read(), dtype=np.uint8)

    color_lut = np.array([class_color(i) for i in range(256)])
    test_array= color_lut[byte_array[0:11]]

    
    img_pre = color_to_image(test_array)
    
    SABV(test_array);

    img = color_to_image(test_array)

    cv2.imshow("img-pre", img_pre)
    cv2.imshow("img", img)
    cv2.waitKey(0)
        

    pass;




