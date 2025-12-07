import time
import numpy as np
import os
import cv2
import math
from hilbertcurve.hilbertcurve import HilbertCurve

from memory_profiler import profile


class SignatureAgnosticBinaryVisualizer:
    def __init__(self, N=5, sample=0.05):
        """
        Initialize the SABV class with parameters.
        
        Args:
            N (int): Window size for neighborhood analysis
            sample (float): Sampling rate for fuzzy domain
        """
        self.N = N
        self.sample = sample
        self.fuzzy_domain = np.arange(0, 1, sample)
        self._precompute_membership_functions()
        self.image_size = (512, 512)
        self.inner_hilbert = self.get_points(9, 2) # because order 9 hilbert

    @staticmethod
    def points_to_order(points_count : int):
        return int(math.log2(math.sqrt(points_count)))

    @staticmethod
    def get_points(p, n):
        if p == 0:
            return np.array([0, 0]).reshape(1, 2)
    
        hilbert_curve = HilbertCurve(p, n)
        points = np.array(hilbert_curve.points_from_distances(
            list(range(2 ** (p * n)))))
        return points
        
        
    @staticmethod
    def class_color(byte):
        """
        Classify bytes into color categories.
        
        Args:
            byte (int): Byte value (0-255)
            
        Returns:
            tuple: RGB color tuple
        """
        if byte == 0:
            return (200, 200, 200)
        elif byte == 255:
            return (255, 255, 0)
        elif 1 <= byte <= 31 or byte == 127:
            return (0, 255, 0)
        elif 32 <= byte <= 126:
            return (0, 0, 255)
        else:
            return (255, 0, 0)
    
    # Membership functions
    @staticmethod
    def u_diff(x):
        """Difference membership function"""
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask1 = (x >= 0) & (x <= 0.2)
        mask2 = (x > 0.2) & (x <= 0.4)
        result[mask1] = 1.0
        result[mask2] = 5 * (0.4 - x[mask2])
        return result

    @staticmethod
    def u_similar(x):
        """Similarity membership function"""
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask1 = (x > 0.2) & (x <= 0.4)
        mask2 = (x > 0.4) & (x <= 0.6)
        mask3 = (x > 0.6) & (x <= 0.8)
        result[mask1] = 5 * (x[mask1] - 0.2)
        result[mask2] = 1.0
        result[mask3] = 5 * (0.8 - x[mask3])
        return result

    @staticmethod
    def u_same(x):
        """Same membership function"""
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask1 = (x > 0.6) & (x <= 0.8)
        mask2 = (x > 0.8)
        result[mask1] = 5 * (x[mask1] - 0.6)
        result[mask2] = 1.0
        return result

    @staticmethod
    def u_light(x):
        """Light intensity membership function"""
        return SignatureAgnosticBinaryVisualizer.u_diff(x)

    @staticmethod
    def u_medium(x):
        """Medium intensity membership function"""
        return SignatureAgnosticBinaryVisualizer.u_similar(x)

    @staticmethod
    def u_dark(x):
        """Dark intensity membership function"""
        return SignatureAgnosticBinaryVisualizer.u_same(x)

    @staticmethod
    def clamp_crisp(x):
        x = np.asarray(x)
        result = np.zeros_like(x)
        mask1 = (x >= 0) & (x <= 0.2)
        mask2 = (x > 0.2) & (x <= 0.65)
        mask3 = (x > 0.65) & (x <= 1)
        result[mask1] = 0.2
        result[mask2] = x[mask2]
        result[mask3] = 1        
        return result
    
    def _precompute_membership_functions(self):
        """
        Precompute membership functions for the entire fuzzy domain.
        This is done once during initialization for performance.
        """
        self.u_light_domain = self.u_light(self.fuzzy_domain).astype(np.float16)
        self.u_medium_domain = self.u_medium(self.fuzzy_domain).astype(np.float16)
        self.u_dark_domain = self.u_dark(self.fuzzy_domain).astype(np.float16)

    def _fuzzy_inference_system(self, current_color, left_side, right_side):
        """
        Perform fuzzy inference to determine brightness adjustment.
        
        Args:
            current_color: Current color being processed
            left_side: Colors to the left of current color
            right_side: Colors to the right of current color
            
        Returns:
            float: Crisp value from fuzzy inference
        """        
        # Calculate similarities with left and right neighborhoods
        left_similarity = np.mean(np.all(left_side == current_color, axis=1)) if len(left_side) > 0 else 0
        right_similarity = np.mean(np.all(right_side == current_color, axis=1)) if len(right_side) > 0 else 0
        
        # Calculate fire strengths
        diff_fire_strength_l = self.u_diff(left_similarity)
        similar_fire_strength_l = self.u_similar(left_similarity)
        same_fire_strength_l = self.u_same(left_similarity)
        
        diff_fire_strength_r = self.u_diff(right_similarity)
        similar_fire_strength_r = self.u_similar(right_similarity)
        same_fire_strength_r = self.u_same(right_similarity)

        # Vectorized aggregation using precomputed membership functions
        aggregate_terms = [
            np.minimum(self.u_light_domain, diff_fire_strength_l),
            np.minimum(self.u_light_domain, diff_fire_strength_r),
            np.minimum(self.u_medium_domain, similar_fire_strength_l),
            np.minimum(self.u_medium_domain, similar_fire_strength_r),
            np.minimum(self.u_dark_domain, same_fire_strength_l),
            np.minimum(self.u_dark_domain, same_fire_strength_r)
        ]        
        aggregate_function = np.maximum.reduce(aggregate_terms)
        
        # Vectorized centroid calculation
        weighted_domain = aggregate_function * self.fuzzy_domain * self.sample
        sum_weights = np.sum(aggregate_function * self.sample)
        
        crisp_value = np.sum(weighted_domain) / sum_weights if sum_weights != 0 else 0
        return crisp_value

    def BinaryVisualizer(self, color_array):
        """
        Main processing method - applies signature-agnostic binary visualization.
        
        Args:
            color_array (numpy.ndarray): Array of colors to process
            
        Returns:
            numpy.ndarray: Processed color array
        """
        new_list = np.zeros(color_array.shape, dtype=np.uint8)
        print(color_array.shape) 
        for i in range(len(color_array)):
            left_side = color_array[max(0, i - self.N):i]
            right_side = color_array[i + 1:min(len(color_array), i + self.N + 1)]

            brightness_index = 1 - self._fuzzy_inference_system(
                color_array[i], left_side, right_side
            )
            new_list[i] = brightness_index * color_array[i]
            
        return new_list


    @profile
    def BinaryVisualizer_v(self, color_array):
        """
        Main processing method - applies signature-agnostic binary visualization. (FASTER)
        Args:
            color_array (numpy.ndarray): Array of colors to process
            
        Returns:
            numpy.ndarray: Processed color array
        """
        
        M = len(color_array)
        left_matches = np.zeros(M, dtype=np.uint8)
        left_counts = np.zeros(M, dtype=np.uint8)
        
        right_matches = np.zeros(M, dtype=np.uint8)
        right_counts = np.zeros(M, dtype=np.uint8)        
        
        for k in range(1, 1 + self.N):
            matches = np.all(color_array[k:] == color_array[:-k], axis=1)
            left_matches[k:] += matches
            left_counts[k:]  += 1
            
            right_matches[:-k] += matches
            right_counts[:-k]  += 1
                    
        left_similarity = np.divide(left_matches, left_counts, out=np.zeros(M), where=left_counts!=0, dtype=np.float16)
        right_similarity = np.divide(right_matches, right_counts, out=np.zeros(M), where=right_counts!=0, dtype=np.float16)

        diff_fire_strength_l    = self.u_diff(left_similarity)
        similar_fire_strength_l = self.u_similar(left_similarity)
        same_fire_strength_l    = self.u_same(left_similarity)
        
        diff_fire_strength_r    = self.u_diff(right_similarity)
        similar_fire_strength_r = self.u_similar(right_similarity)
        same_fire_strength_r    = self.u_same(right_similarity)
        
        D = self.u_light_domain.shape[0]
        aggregate_function = np.zeros((M, D), dtype=np.float16)

        rules = [
            (diff_fire_strength_l,    self.u_light_domain),
            (diff_fire_strength_r,    self.u_light_domain),
            (similar_fire_strength_l, self.u_medium_domain),
            (similar_fire_strength_r, self.u_medium_domain),
            (same_fire_strength_l,    self.u_dark_domain),
            (same_fire_strength_r,    self.u_dark_domain)
        ]
        for strength, domain in rules:
            np.maximum(aggregate_function, np.minimum(strength[:, None], domain[None, :]), out=aggregate_function)


        # Defuzzification (Centroid)
        # We compute the weighted average across the Domain axis (axis 1)
        
        # Numerator: Sum(Area * x) -> Sum over axis 1
        # self.fuzzy_domain must be broadcasted to (1, D) or simply (D,) works with (M, D)
        weighted_sum = np.sum(aggregate_function * self.fuzzy_domain * self.sample, axis=1)
        
        # Denominator: Sum(Area) -> Sum over axis 1
        sum_weights = np.sum(aggregate_function * self.sample, axis=1)
        
        # 6. Final Division (Handle division by zero)
        # Result shape: (M,)
        crisp_values = np.divide(
            weighted_sum, 
            sum_weights, 
            out=np.zeros_like(weighted_sum), 
            where=sum_weights != 0
        )
        crisp_values = 1 - crisp_values
        crisp_values = self.clamp_crisp(crisp_values)
        
        new_list = color_array * crisp_values[:, None].astype(np.float32)
        return new_list.astype(np.uint8)
    
    def process_file(self, file_path):
        """
        Convenience method to process a file directly.
        
        Args:
            file_path (str): Path to the file to process
            
        Returns:
            tuple: (processed_color_array, execution_time)
        """
        # Read and classify bytes
        with open(file_path, 'rb') as file:
            byte_array = np.frombuffer(file.read(), dtype=np.uint8)

        total_bytes = len(byte_array)
        
        image_chunk_size = tuple()
        if total_bytes < 256 * 1024:
            image_chunk_size = (512, 512)
        elif  256 * 1024 <= total_bytes:
            image_chunk_size = (256, 256)

        chunk_count = int(np.prod(self.image_size) / np.prod(image_chunk_size))
        outer_hilbert = self.get_points(self.points_to_order(chunk_count), 2) * image_chunk_size[0]

        max_bytes = chunk_count * np.prod(self.image_size)
        if total_bytes < max_bytes:
            byte_array = np.pad(
                byte_array, (0, max_bytes - len(byte_array)), mode="constant", constant_values=0)
        else:
            byte_array = byte_array[0:max_bytes]    
            
        print(f"total_bytes {total_bytes}, chunk_side {image_chunk_size}, max_bytes {max_bytes}")
            
        color_lut = np.array([self.class_color(i) for i in range(256)])
        colored_byte_array = color_lut[byte_array]
        
        processed_array = self.BinaryVisualizer_v(colored_byte_array)
                                
        full_image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        image_pixel_count = np.prod(self.image_size);

        for i, (ox, oy) in enumerate(outer_hilbert):
            start_index = i * image_pixel_count
            end_index = (i + 1) * image_pixel_count
            
            current_chunk = processed_array[start_index:end_index]
        
            inner_hilbert_x = self.inner_hilbert[:, 0]
            inner_hilbert_y = self.inner_hilbert[:, 1]

            chunk_image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            
            chunk_image[inner_hilbert_y, inner_hilbert_x] = current_chunk

            resized_chunk = cv2.resize(chunk_image, image_chunk_size, interpolation=cv2.INTER_LINEAR)
            full_image[oy:oy + image_chunk_size[0], ox:ox + image_chunk_size[1]] = resized_chunk
        
        return full_image

# Example usage
if __name__ == "__main__":
    # Create SABV instance
    sabv = SignatureAgnosticBinaryVisualizer(N=3)
    
    file_path = os.getcwd() + "/PE-files/544.exe" 
    start = time.perf_counter()
    img = sabv.process_file(file_path)
    end = time.perf_counter()

    print(f"Execution time: {end - start:.4f} seconds")
    
    cv2.imshow("img", img)
    cv2.waitKey(0)
