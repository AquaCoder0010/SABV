from joblib import Parallel, delayed
import time
import numpy as np
import os
import cv2
import math
from hilbertcurve.hilbertcurve import HilbertCurve

# Signature Agnostic Binary Visualizer 
class SignatureAgnosticBinaryVisualizer:
    def __init__(self, FIS_ENABLED=False, N=None, sample=None, FIS_THREADING_ENABLED=False):
        """
        Initialize the SABV class with parameters.
        
        Args:
            N (int): Window size for neighborhood analysis
            sample (float): Sampling rate for fuzzy domain
        """
        self.FIS_ENABLED=FIS_ENABLED
        self.FIS_THREADING_ENABLED=None
        self.N = None
        self.sample = None
        
        if self.FIS_ENABLED == False and FIS_THREADING_ENABLED == True:
            print("Threading can't be true while FIS disabled!")
            self.FIS_THREADING_ENABLED=False
        if self.FIS_ENABLED == False and N != None:
            print(f"Similarity Space (N) can't be type:{type(N)} while FIS disabled!")
            self.N = None
        if self.FIS_ENABLED == False and sample != None:
            print(f"sample for fuzzy domain (sample) can't be type:{type(sample)} while FIS disabled!")
            self.sample = None
        else:
            self.N = N
            self.sample = sample
            self.fuzzy_domain = np.arange(0, 1, sample)
            self._precompute_membership_functions()
            self.FIS_THREADING_ENABLED=FIS_THREADING_ENABLED
                    
        self.image_size = (512, 512)
        self.inner_hilbert = self.get_points(9, 2) # because order 9 hilbert
        self.user_def_color_scheme = None
        
    def set_color_scheme(self, func):
        # Check if the function is callable
        if not callable(func):
            raise TypeError(
                f"Invalid lambda function, parameter is type: {type(func)}"
            )
        DUMMY_VARIABLE = 0
        # Check if the function returns a tuple
        result = func(DUMMY_VARIABLE)
        if not isinstance(result, tuple):
            raise ValueError(
                "Invalid output definition defined in custom function! "
                f"Expected tuple, got {type(result)}"
            )
        self.user_def_color_scheme = func
        
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

        weighted_sum = np.sum(aggregate_function * self.fuzzy_domain * self.sample, axis=1)
        
        sum_weights = np.sum(aggregate_function * self.sample, axis=1)
        
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


    def BinaryVisualizer_vt2(self, color_array):        
        def process(start_index, end_index):
            M = end_index - start_index + 1
            left_matches = np.zeros(M, dtype=np.uint8)
            left_counts = np.zeros(M, dtype=np.uint8)
            
            right_matches = np.zeros(M, dtype=np.uint8)
            right_counts = np.zeros(M, dtype=np.uint8)        
            
            for k in range(1, 1 + self.N):
                start_slice = start_index + k
                end_slice = end_index - k

                matches = np.all(color_array[start_slice: (end_index + 1)] == color_array[start_index:(end_slice + 1)], axis=1)

                l_start = start_slice - start_index
                l_end   = (end_index + 1) - start_index
                
                r_start = 0 
                r_end   = (end_slice + 1) - start_index
                
                left_matches[l_start:l_end] += matches
                left_counts[l_start:l_end]  += 1
    
                right_matches[r_start:r_end] += matches
                right_counts[r_start:r_end]  += 1
                
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
                
            weighted_sum = np.sum(aggregate_function * self.fuzzy_domain * self.sample, axis=1)
            sum_weights = np.sum(aggregate_function * self.sample, axis=1)
            
            crisp_values = np.divide(
                weighted_sum, 
                sum_weights, 
                out=np.zeros_like(weighted_sum), 
                where=sum_weights != 0
            )
            crisp_values = 1 - crisp_values
            crisp_values = self.clamp_crisp(crisp_values)

            return crisp_values

        core_count = os.cpu_count()
        M = len(color_array)
        chunk_size = math.ceil(M / core_count);        
        tasks = [];
        for i in range(core_count):    
            start_i = i * chunk_size
            end_i = (i + 1) * chunk_size - 1
            
            tasks.append(delayed(process)(
                start_i,
                end_i,
            ))

        results = Parallel(n_jobs=core_count)(tasks)

        new_list = color_array * np.concatenate(results)[:, None].astype(np.float32)
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

        color_lut = None
        if self.user_def_color_scheme != None:
            color_lut = np.array([self.user_def_color_scheme(i) for i in range(256)])
        else:
            color_lut = np.array([self.class_color(i) for i in range(256)])

        colored_byte_array = color_lut[byte_array]

        if self.FIS_ENABLED == True:
            core_count = os.cpu_count()
            if core_count == 1 and self.FIS_THREADING_ENABLED:
                print(f"Core count is {core_count}, disabling FIS_THREADING")
                processed_array = self.BinaryVisualizer_v(colored_byte_array)
            else:
                processed_array = self.BinaryVisualizer_vt2(colored_byte_array)
        else:
            processed_array = colored_byte_array
                                
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
    sabv_fis = SignatureAgnosticBinaryVisualizer(FIS_ENABLED=True, N=3, sample=0.05, FIS_THREADING_ENABLED=True)
    sabv = SignatureAgnosticBinaryVisualizer()

    
    
    file_path = os.getcwd() + "/PE-files/546.exe" 

    print(f"No FIS")
    start = time.perf_counter()
    img = sabv.process_file(file_path)
    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")

    print(f"No FIS, custom color scheme")
    def custom(byte):
        """
        Classify a byte (0–255) into one of 16 color bins:
        00, 11, 22, ..., EE, FF.
        
        The high nibble (byte >> 4) determines the class.
        """        
        # High nibble: 0–15
        nibble = byte >> 4
        
        # Mapping nibble 0–15 to RGB values
        color_map = {
            0x0: (0,   0,   0),
            0x1: (128, 0,   0),
            0x2: (154, 99,  36),
            0x3: (128, 128, 0),
            0x4: (70,  153, 144),
            0x5: (0,   0,   117),
            0x6: (230, 25,  75),
            0x7: (245, 130, 49),
            0x8: (255, 225, 25),
            0x9: (191, 239, 69),
            0xA: (60,  180, 75),
            0xB: (66,  212, 244),
            0xC: (67,  99,  216),
            0xD: (145, 30,  180),
            0xE: (240, 50,  230),
            0xF: (255, 255, 255),
        }
        
        return color_map[nibble]    
    sabv.set_color_scheme(custom)

    start = time.perf_counter()
    img_2 = sabv.process_file(file_path)
    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")

    

    print(f"with FIS")
    start = time.perf_counter()
    img_3 = sabv_fis.process_file(file_path)
    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")

    
    cv2.imshow("img", img)
    cv2.imshow("img_custom", img_2)
    cv2.imshow("img_fis", img_3)
    cv2.waitKey(0)

    cv2.imwrite("sabv-no-FIS.png",img);
    cv2.imwrite("sabv-FIS.png",img_2);
    
    
    
SABV = SignatureAgnosticBinaryVisualizer
