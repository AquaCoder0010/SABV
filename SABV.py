from joblib import Parallel, delayed
import time
import numpy as np
import os
import cv2
import math
from hilbertcurve.hilbertcurve import HilbertCurve

from memory_profiler import profile

from enum import Enum

class Options(Enum):
    FIS_ENABLED = 1
    FIS_DISABLED = 2


class SignatureAgnosticBinaryVisualizer:
    def __init__(self, N=0, option=Options.FIS_DISABLED, sample=0.05):
        """
        Initialize the SABV class with parameters.
        
        Args:
            N (int): Window size for neighborhood analysis
            sample (float): Sampling rate for fuzzy domain
        """
        self.N = 2
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

        
    def BinaryVisualizer_vt2(self, color_array):
        
        def process(start_index, end_index):
            assert(end_index > start_index), "invalid indexing"

            M = end_index - start_index + 1

            print("PROCESS")
            print(f"M:{M}")
            left_matches = np.zeros(M, dtype=np.uint8)
            left_counts = np.zeros(M, dtype=np.uint8)
            
            right_matches = np.zeros(M, dtype=np.uint8)
            right_counts = np.zeros(M, dtype=np.uint8)        
            
            for k in range(1, 1 + self.N):
                start_slice = start_index + k
                end_slice = end_index - k

                print(f"start_slice:{start_slice}, start_index: {start_index}, end_slice: {end_slice}, end_index: {end_index}")
                matches = np.all(color_array[start_slice: (end_index + 1)] == color_array[start_index:(end_slice + 1)], axis=1)
                print(f"size of matches :  {len(matches)}, shape : {matches.shape}")
                print(f"size of left_matches : {len(left_matches)}")
                print(f"size of left_counts : {len(left_counts)}")
                print(f"size of right_matches : {len(right_matches)}")
                print(f"size of right_counts : {len(right_counts)}")
                
                
                
                left_matches[start_slice: (end_index + 1)] += matches
                left_counts[start_slice: (end_index + 1)]  += 1
    
                right_matches[start_index:(end_slice + 1)] += matches
                right_counts[start_index:(end_slice + 1)]  += 1

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

            # Defuzzification (using einsum for efficiency)
            domain_weights = self.fuzzy_domain * self.sample
            weighted_sum = np.einsum('ij,j->i', aggregate_function, domain_weights)
            sum_weights = np.einsum('ij,j->i', aggregate_function, self.sample)

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
            print(start_i, end_i)

        results = Parallel(n_jobs=4)(tasks)
        return np.concatenate(results)
    
    def BinaryVisualizer_vt(self, color_array, core_count):
        """
        Main processing method, parallelized by dividing the color_array into chunks 
        based on the number of cores (core_count / 2).
        """
        M = len(color_array)
        N = self.N # The maximum shift distance (from the k-loop)
        
        # 1. Determine parallel settings
        #n_jobs = max(1, core_count // 2)
        
        # Calculate chunk size, ensuring each core gets roughly the same amount of data
        chunk_size = math.ceil(M / n_jobs)
    
        # --- Worker Function to process a single chunk ---
        def process_chunk(chunk_data, start_index, end_index, M_total, N_shift):
            
            # M_chunk is the length of the data we actually care about (without padding)
            # It handles the potentially smaller last chunk
            M_chunk = end_index - start_index
            
            # 1. Sliding Window (k-loop)
            
            # Initialization size is M_chunk
            left_matches = np.zeros(M_chunk, dtype=np.uint8)
            left_counts = np.zeros(M_chunk, dtype=np.uint8)
            right_matches = np.zeros(M_chunk, dtype=np.uint8)
            right_counts = np.zeros(M_chunk, dtype=np.uint8)
        
            # The k-loop uses the full chunk_data (which includes the padding)
            # The comparison slice must be done on the indices *relative to the chunk_data*
        
            # The results of matches are only relevant for the length M_total - k, 
            # so we need careful indexing

            # 1000
            # 5
            # 200, 400, 600, 800, 1000
            # 0:199, 200:399, 400:599, 600:799, 800:999
            
            for k in range(1, 1 + N_shift):
                
                # --- Matches calculation (uses padding) ---
                # matches is calculated over the common, shifted region of the chunk_data
                # Length of matches = len(chunk_data) - k
                matches = np.all(chunk_data[k:] == chunk_data[:-k], axis=1)
                
                # --- Left Matches/Counts Update ---
                # Affects indices k:M_chunk (relative to the unpadded output)
                # The indices for the update start at (k - padding_left), but since
                # padding_left = N_shift and N_shift >= k, the start index is 0 or k-N_shift
                
                # Left side updates (LHS of color_array) use chunk_data[k:] 
                # which corresponds to indices [0 + k: M_chunk + N_shift - k] in the output space
                
                # We map the matches array (len(chunk_data) - k) to the output arrays (len M_chunk)
                
                # Indices relative to the output array (0 to M_chunk-1)
                # Start of affected region: max(0, k - N_shift)
                # End of affected region: M_chunk 
                
                # Start index in matches array: k - N_shift (or 0 if negative)
                match_start_idx = max(0, k - N_shift)
                
                # End index in matches array: The end of the M_chunk area
                match_end_idx = match_start_idx + M_chunk - max(0, k - N_shift)
                
                # Ensure slicing doesn't exceed array bounds
                update_len = M_chunk - max(0, k - N_shift)
                
                if update_len > 0 and (k - start_index < M_total):
                
                    # left_matches[k - N_shift:] are updated by matches[k - N_shift:]
                    # Where k-N_shift is the offset into the unpadded output buffer (left_matches)
                    # and the matches buffer (matches).
                    
                    # Slice indices:
                    # Output slice: [max(0, k - N_shift): M_chunk]
                    # Matches slice: [max(0, k - N_shift): max(0, k - N_shift) + (M_chunk - max(0, k - N_shift))]
                    
                    left_matches[max(0, k - N_shift):] += matches[max(0, k - N_shift):M_chunk + N_shift - k]
                    left_counts[max(0, k - N_shift):] += 1
                    
                    # --- Right Matches/Counts Update ---
                    # Affects indices 0:M_chunk-k (relative to the unpadded output)
                    # output slice: [0 : M_chunk - k]
                    # matches slice: [N_shift : N_shift + M_chunk - k]
                    
                    # Right side updates (RHS of color_array) use chunk_data[:-k]
                    # The updates stop at M_chunk - k 
                    if M_chunk - k > 0:
                        right_matches[:M_chunk - k] += matches[N_shift : N_shift + M_chunk - k]
                        right_counts[:M_chunk - k] += 1
                        
                    # 2. Similarity and Fire Strength Calculation (Localized)
        
                    # Calculate similarities
                    left_similarity = np.divide(left_matches, left_counts, out=np.zeros(M_chunk), where=left_counts!=0, dtype=np.float16)
                    right_similarity = np.divide(right_matches, right_counts, out=np.zeros(M_chunk), where=right_counts!=0, dtype=np.float16)
                    
                    # Calculate fire strengths (assuming u_diff/u_similar/u_same are vectorized)
                    diff_fire_strength_l = self.u_diff(left_similarity)
                    similar_fire_strength_l = self.u_similar(left_similarity)
                    same_fire_strength_l = self.u_same(left_similarity)
                    
                    diff_fire_strength_r = self.u_diff(right_similarity)
                    similar_fire_strength_r = self.u_similar(right_similarity)
                    same_fire_strength_r = self.u_same(right_similarity)
                    
                    # 3. Fuzzy Aggregation and Defuzzification (Localized)
                    
                    D = self.u_light_domain.shape[0]
                    aggregate_function = np.zeros((M_chunk, D), dtype=np.float16)
                    
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

                    # Defuzzification (using einsum for efficiency)
                    domain_weights = self.fuzzy_domain * self.sample
                    weighted_sum = np.einsum('ij,j->i', aggregate_function, domain_weights)
                    sum_weights = np.einsum('ij,j->i', aggregate_function, self.sample)
                        
                    # Final Division
                    crisp_values = np.divide(
                        weighted_sum, 
                        sum_weights, 
                        out=np.zeros_like(weighted_sum), 
                        where=sum_weights != 0
                    )
                    crisp_values = 1 - crisp_values
                    crisp_values = self.clamp_crisp(crisp_values)
        
                    # Return the resulting crisp values for merging
                    return crisp_values

        # --- Main Loop Setup (Splitting Data) ---
        
        # Padding: We need N elements before and N elements after each chunk for the k-loop calculation
        # Only the first and last chunks need special padding handling.
        tasks = []
    
        for i in range(n_jobs):
            # Calculate the pure start and end index (where output will be merged)
            start_index = i * chunk_size
            end_index = min((i + 1) * chunk_size, M)
            
            # Determine the padded start and end indices for the input data slice
            # Left boundary padding (N elements from the previous chunk, or 0 if first chunk)
            data_start = max(0, start_index - N)
            
            # Right boundary padding (N elements from the next chunk, or M if last chunk)
            data_end = min(M, end_index + N)
            
            # The slice of the original color_array to be processed
            chunk_data = color_array[data_start:data_end]
            
            # The padding size on the left side (N for inner chunks, or less for the first chunk)
            padding_left = start_index - data_start 
        
            # The worker function needs to know the indices of the *unpadded* output data
            tasks.append(delayed(process_chunk)(
                chunk_data,
                start_index,
                end_index,
                M,
                N
            ))

        # Execute all tasks in parallel
        results = Parallel(n_jobs=n_jobs)(tasks)
    
        # 4. Merge Results
        
        # Concatenate all the resulting crisp_values (M, ) back together
        merged_crisp_values = np.concatenate(results)
        
        # Final multiplication (sequential, fast, and uses explicit dtypes)
        new_list = color_array * merged_crisp_values[:, None].astype(np.float32)
    
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

        core_count = os.cpu_count()
        if core_count == 1:
            processed_array = self.BinaryVisualizer_v(colored_byte_array)
        else:
            processed_array = self.BinaryVisualizer_vt2(colored_byte_array)
            
                                
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
    sabv = SignatureAgnosticBinaryVisualizer(N=3, option=Options.FIS_ENABLED)
    
    file_path = os.getcwd() + "/PE-files/uwu.exe" 
    start = time.perf_counter()
    img = sabv.process_file(file_path)
    end = time.perf_counter()

    print(f"Execution time: {end - start:.4f} seconds")
    
    cv2.imshow("img", img)
    cv2.waitKey(0)
