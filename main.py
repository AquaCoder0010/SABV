import time

from hilbertcurve.hilbertcurve import HilbertCurve

import numpy as np
import os
import cv2

import math
import random

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
    x = np.asarray(x)
    result = np.zeros_like(x)
    mask1 = (x >= 0) & (x <= 0.2)
    mask2 = (x > 0.2) & (x <= 0.4)
    result[mask1] = 1.0
    result[mask2] = 5 * (0.4 - x[mask2])
    return result

def u_similar(x):
    x = np.asarray(x)
    result = np.zeros_like(x)
    mask1 = (x > 0.2) & (x <= 0.4)
    mask2 = (x > 0.4) & (x <= 0.6)
    mask3 = (x > 0.6) & (x <= 0.8)
    result[mask1] = 5 * (x[mask1] - 0.2)
    result[mask2] = 1.0
    result[mask3] = 5 * (0.8 - x[mask3])
    return result

def u_same(x):
    x = np.asarray(x)
    result = np.zeros_like(x)
    mask1 = (x > 0.6) & (x <= 0.8)
    mask2 = (x > 0.8)
    result[mask1] = 5 * (x[mask1] - 0.6)
    result[mask2] = 1.0
    return result
def u_light(x):
    return u_diff(x)

def u_medium(x):
    return u_similar(x) 

def u_dark(x):
    return u_same(x) 


# Precompute membership functions for entire domain (do this once outside the loop)
def precompute_membership_functions(fuzzy_domain):
    u_light_domain = u_light(fuzzy_domain)
    u_medium_domain = u_medium(fuzzy_domain) 
    u_dark_domain = u_dark(fuzzy_domain)
    return u_light_domain, u_medium_domain, u_dark_domain


def fuzzy_inference_system_optimized(current_color, left_side, right_side, fuzzy_domain, sample,
                                   u_light_domain, u_medium_domain, u_dark_domain):
    left_side = np.array(left_side)
    right_side = np.array(right_side)
    current_color = np.array(current_color)
    
    left_similarity = np.mean(np.all(left_side == current_color, axis=1)) if len(left_side) > 0 else 0
    right_similarity = np.mean(np.all(right_side == current_color, axis=1)) if len(right_side) > 0 else 0
    
    # Calculate fire strengths
    diff_fire_strength_l = u_diff(left_similarity)
    similar_fire_strength_l = u_similar(left_similarity)
    same_fire_strength_l = u_same(left_similarity)

    diff_fire_strength_r = u_diff(right_similarity)
    similar_fire_strength_r = u_similar(right_similarity)
    same_fire_strength_r = u_same(right_similarity)

    # Vectorized aggregation using precomputed membership functions
    aggregate_terms = [
        np.minimum(u_light_domain, diff_fire_strength_l),
        np.minimum(u_light_domain, diff_fire_strength_r),
        np.minimum(u_medium_domain, similar_fire_strength_l),
        np.minimum(u_medium_domain, similar_fire_strength_r),
        np.minimum(u_dark_domain, same_fire_strength_l),
        np.minimum(u_dark_domain, same_fire_strength_r)
    ]
    
    aggregate_function = np.maximum.reduce(aggregate_terms)
    
    # Vectorized centroid calculation
    weighted_domain = aggregate_function * fuzzy_domain * sample
    sum_weights = np.sum(aggregate_function * sample)
    
    crisp_value = np.sum(weighted_domain) / sum_weights if sum_weights != 0 else 0
    return crisp_value


# Signature Agnostic Binary Visualizer
def SABV(color_list, N=5):
    new_list = np.zeros(color_list.shape, dtype=np.uint8)

    color_list_length = len(color_list)

    SAMPLE = 0.01
    fuzzy_domain = np.arange(0, 1, SAMPLE)
    
    # Precompute once outside the loop
    u_light_domain, u_medium_domain, u_dark_domain = precompute_membership_functions(fuzzy_domain)

    # Convert color_list to numpy array for faster operations
    color_array = np.array(color_list)

    for i in range(len(color_array)):
        left_side = color_array[max(0, i - N):i]
        right_side = color_array[i + 1:min(len(color_array), i + N + 1)]

        brightness_index = 1 - fuzzy_inference_system_optimized(
            color_array[i], left_side, right_side, fuzzy_domain, SAMPLE,
            u_light_domain, u_medium_domain, u_dark_domain
        )
        new_list[i] = brightness_index * color_array[i]
        
    return new_list

if __name__ == "__main__":
    # test path
    file_path = os.getcwd() + "/PE-files/546.exe"    
    with open(file_path, 'rb') as file:
        byte_array = np.frombuffer(file.read(), dtype=np.uint8)

    color_lut = np.array([class_color(i) for i in range(256)])
    colored_byte_array = color_lut[byte_array]

    start = time.perf_counter()
    colored_byte_array = SABV(colored_byte_array)
    
    end = time.perf_counter()

    # pass it through the PE to Image Conversion pipeline
    print(colored_byte_array)
    print(f"Execution time: {end - start:.4f} seconds")

    print(len(colored_byte_array))    
    pass;
