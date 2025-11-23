import time
import numpy as np
import os
import cv2
import math
from hilbertcurve.hilbertcurve import HilbertCurve

class SignatureAgnosticBinaryVisualizer:
    def __init__(self, N=5, sample=0.01):
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
            return (128, 128, 128) 
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

    def _precompute_membership_functions(self):
        """
        Precompute membership functions for the entire fuzzy domain.
        This is done once during initialization for performance.
        """
        self.u_light_domain = self.u_light(self.fuzzy_domain)
        self.u_medium_domain = self.u_medium(self.fuzzy_domain) 
        self.u_dark_domain = self.u_dark(self.fuzzy_domain)

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

    def BinaryVisualizer(self, color_list):
        """
        Main processing method - applies signature-agnostic binary visualization.
        
        Args:
            color_list (numpy.ndarray): Array of colors to process
            
        Returns:
            numpy.ndarray: Processed color array
        """
        # (1559552, 3)
        print(color_list.shape)

        new_list = np.zeros(color_list.shape, dtype=np.uint8)
        color_array = np.array(color_list)

        for i in range(len(color_array)):
            left_side = color_array[max(0, i - self.N):i]
            right_side = color_array[i + 1:min(len(color_array), i + self.N + 1)]

            brightness_index = 1 - self._fuzzy_inference_system(
                color_array[i], left_side, right_side
            )
            new_list[i] = brightness_index * color_array[i]
            
        return new_list

    
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
            
        color_lut = np.array([self.class_color(i) for i in range(256)])
        colored_byte_array = color_lut[byte_array]
        
        # Process the array
        start = time.perf_counter()
        processed_array = self.BinaryVisualizer(colored_byte_array)
        end = time.perf_counter()

        execution_time = end - start
        return processed_array, execution_time

# Example usage and main function
if __name__ == "__main__":
    # Create SABV instance
    sabv = SignatureAgnosticBinaryVisualizer(N=3)
    
    # Test path
    file_path = os.getcwd() + "/PE-files/544.exe"    
    
    # Process file
    processed_array, exec_time = sabv.process_file(file_path)
    
    print(processed_array)
    print(f"Execution time: {exec_time:.4f} seconds")
    print(f"Processed {len(processed_array)} bytes")
    
    # Optional: Create and save image
    # image = sabv.color_to_image(processed_array[:11])  # First 11 colors for demo
    # cv2.imwrite("output.png", image)
