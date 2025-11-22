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


def u_diff(x):
    if 0 <= x <= 0.2:
        return 1.0
    elif 0.2 < x <= 0.4:
        return 5 * (0.4 - x)
    return 0
    
def u_similar(x):
    if 0 <= x <= 0.2:
        return 0
    elif 0.2 < x <= 0.4:
        return 5 * (x - 0.2)
    elif 0.4 < x <= 0.6:
        return 1
    elif 0.6 < x <= 0.8:
        return 5 * (0.8 - x)
    return 0

def u_same(x):
    if 0 <= x <= 0.6:
        return 0
    elif 0.6 < x <= 0.8:
        return 5 * (x - 0.6)
    return 1

def u_light(x):
    return u_diff(x)

def u_medium(x):
    return u_similar(x) 

def u_dark(x):
    return u_same(x) 


def fuzzy_inference_system(current_color, left_side, right_side):    
    SAMPLE = 0.001


    left_similarity = (
        np.sum([int(np.array_equal(current_color, color_byte)) for color_byte in left_side]) / len(left_side)
        if len(left_side) > 0 else 0
    )
    
    right_similarity = (
        np.sum([int(np.array_equal(current_color, color_byte)) for color_byte in right_side]) / len(right_side)
        if len(right_side) > 0 else 0
    )
    
    
    diff_fire_strength_l = u_diff(left_similarity);
    similar_fire_strength_l = u_similar(left_similarity);
    same_fire_strength_l = u_same(left_similarity);

    diff_fire_strength_r = u_diff(right_similarity);
    similar_fire_strength_r = u_similar(right_similarity);
    same_fire_strength_r = u_same(right_similarity);

    aggregate_function_domain = [x for x in np.arange(0, 1, SAMPLE)]    
    aggregate_function = np.array([
        max(
            min(u_light(x), diff_fire_strength_l),
            min(u_light(x), diff_fire_strength_r),
            min(u_medium(x), similar_fire_strength_l),
            min(u_medium(x), similar_fire_strength_r),
            min(u_dark(x), same_fire_strength_l),
            min(u_dark(x), same_fire_strength_r),
        ) for x in aggregate_function_domain
    ])
    nominator = np.sum(aggregate_function * aggregate_function_domain * SAMPLE)
    denominator = np.sum(aggregate_function * SAMPLE)

    crisp_value = nominator / denominator;    
    return crisp_value;
    
# Signature Agnostic Binary Visualizer
def SABV(color_list):
    N = 5    
    new_list = np.zeros((11, 3), dtype=np.uint8)

    color_list_length = len(color_list)
    for i in range(len(color_list)):
        left_side = color_list[max(0, i - N):i]
        right_side = color_list[i + 1:min(color_list_length, i + N + 1)]

        brightness_index = 1 - fuzzy_inference_system(color_list[i], left_side, right_side)
        print(brightness_index)
        
        new_list[i] = brightness_index * color_list[i]
        print(new_list[i], color_list[i])
    return new_list
                

if __name__ == "__main__":
    # test path
    file_path = os.getcwd() + "/PE-files/546.exe"    
    with open(file_path, 'rb') as file:
        byte_array = np.frombuffer(file.read(), dtype=np.uint8)

    color_lut = np.array([class_color(i) for i in range(256)])
    test_array= color_lut[byte_array[0:11]]

    
    img_pre = color_to_image(test_array)    
    img = color_to_image(SABV(test_array))
    cv2.imshow("img-pre", img_pre)
    cv2.imshow("img", img)
    cv2.waitKey(0)
        

    pass;




