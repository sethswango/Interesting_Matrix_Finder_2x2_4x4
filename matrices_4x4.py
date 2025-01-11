import numpy as np
import cupy as cp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime
from time import perf_counter

# Constants
VALUE_RANGE = (1, 9)
MATRIX_SIZES = [4]  # Changed to 4x4
NUM_PROCESSES = 4
GPU_BATCH_SIZE = 1000  # Reduced due to larger matrix size
CUDA_MANAGED_MEMORY = True

def matrix_from_index(index, size):
    """Optimized for 4x4 case"""
    if size == 4:
        matrix = np.empty((4, 4), dtype=np.int8)
        for i in range(4):
            for j in range(4):
                matrix[i,j] = (index & 0b1111) % 9 + 1
                index >>= 4
        return matrix
    return None

def check_concatenation_property(A, B, C):
    """Optimized concatenation check for 4x4 matrices"""
    for i in range(4):
        for j in range(4):
            concat = (A[i,j] * 10) + B[i,j]
            if C[i,j] != concat:
                return False
    return True

def get_valid_indices(start, end, size):
    """Pre-compute valid matrix indices for 4x4 matrices"""
    indices = []
    for i in range(start, end):
        matrix = matrix_from_index(i, size)
        if matrix is not None and np.all(matrix >= 1) and np.all(matrix <= 9):
            indices.append(i)
    return np.array(indices, dtype=np.int32)

def process_batch(params):
    """Optimized batch processing for 4x4 matrices"""
    size, start_idx, end_idx = params
    matches = []
    total_B = get_total_combinations(size)
    timings = {'setup': 0, 'gpu_transfer': 0, 'compute': 0, 'check': 0}
    
    # Adjusted batch sizes for 4x4
    PARALLEL_A_COUNT = 256  # Reduced from 4096 due to larger matrices
    GPU_BATCH_SIZE = 1000   # Reduced from 20000
    
    # Get valid indices for both A and B matrices
    valid_A_indices = get_valid_indices(start_idx, end_idx, size)
    valid_B_indices = get_valid_indices(0, total_B, size)
    
    if len(valid_A_indices) == 0 or len(valid_B_indices) == 0:
        return matches
    
    # Pre-generate all valid B matrices at once
    B_matrices = np.stack([matrix_from_index(i, size) for i in valid_B_indices])
    B_matrices_gpu = cp.asarray(B_matrices, dtype=cp.int8)
    
    # Process only valid A matrices
    for idx_A_batch in range(0, len(valid_A_indices), PARALLEL_A_COUNT):
        batch_indices = valid_A_indices[idx_A_batch:idx_A_batch + PARALLEL_A_COUNT]
        batch_size = len(batch_indices)
        
        # Generate only valid A matrices
        A_cpu_batch = np.stack([matrix_from_index(i, size) for i in batch_indices])
        A_gpu = cp.asarray(A_cpu_batch, dtype=cp.int8)
        
        # Process B matrices in smaller chunks
        for batch_start in range(0, len(B_matrices_gpu), GPU_BATCH_SIZE):
            batch_end = min(batch_start + GPU_BATCH_SIZE, len(B_matrices_gpu))
            B_batch = B_matrices_gpu[batch_start:batch_end]
            
            t_compute = perf_counter()
            # Compute matrix multiplications
            C_batch = cp.matmul(
                A_gpu.reshape(batch_size, 1, 4, 4),
                B_batch.reshape(1, -1, 4, 4)
            )
            
            # Transfer results back to CPU
            A_local = cp.asnumpy(A_gpu)
            B_local = cp.asnumpy(B_batch)
            C_local = cp.asnumpy(C_batch)
            timings['compute'] += perf_counter() - t_compute
            
            t_check = perf_counter()
            # Vectorized concatenation check
            concat_check = (A_local.reshape(batch_size, 1, 4, 4) * 10 + 
                          B_local.reshape(1, -1, 4, 4))
            
            matches_mask = (concat_check == C_local)
            matches_mask = matches_mask.all(axis=(2, 3))
            
            # Get indices where matches occurred
            a_indices, b_indices = np.where(matches_mask)
            
            # Add matches with verification
            for a_idx, b_idx in zip(a_indices, b_indices):
                A_candidate = A_local[a_idx]
                B_candidate = B_local[b_idx]
                C_candidate = C_local[a_idx, b_idx]
                
                if verify_match(A_candidate, B_candidate, C_candidate):
                    matches.append({
                        "size": size,
                        "A": A_candidate.copy(),
                        "B": B_candidate.copy(),
                        "C": C_candidate.copy()
                    })
            
            timings['check'] += perf_counter() - t_check
    
    return matches

def store_matches(matches):
    """Use numpy structured arrays for more efficient storage"""
    dtype = np.dtype([
        ('A', np.int8, (4, 4)),
        ('B', np.int8, (4, 4)),
        ('C', np.int8, (4, 4))
    ])
    return np.array(matches, dtype=dtype)

def vectorized_concat_check(A_batch, B_batch, C_batch):
    """Optimize concatenation check with vectorized operations for 4x4"""
    concat = np.multiply(A_batch, 10, dtype=np.int16) + B_batch
    return np.all(concat == C_batch, axis=(2, 3))

def matrix_multiplication_check(A, B):
    """Quick check before full multiplication for 4x4"""
    if np.any(A > 9) or np.any(B > 9):
        return False
    if np.any(A == 0) or np.any(B == 0):
        return False
    return True

def find_special_matrices_exhaustive():
    """Matches base solution exactly but for 4x4"""
    all_matches = []
    
    for size in MATRIX_SIZES:
        total_combinations = get_total_combinations(size)
        print(f"\nSearching {size}x{size} matrices...")
        print(f"Total possible matrices: {total_combinations:,}")
        # Fix integer overflow by using explicit float calculation
        total_checks = float(total_combinations) * float(total_combinations)
        print(f"Total combinations to check: {total_checks:,.0f}")
        print(f"Using {NUM_PROCESSES} parallel processes")
        
        batch_size = total_combinations // NUM_PROCESSES
        batches = [(size, i * batch_size, (i + 1) * batch_size) for i in range(NUM_PROCESSES)]
        if batches[-1][2] < total_combinations:
            batches[-1] = (size, batches[-1][1], total_combinations)
        
        with tqdm(total=total_combinations, desc=f"{size}x{size} matrices") as pbar:
            with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
                futures = [executor.submit(process_batch, batch) for batch in batches]
                
                for future in futures:
                    matches = future.result()
                    all_matches.extend(matches)
                    pbar.update(batch_size)
    
    return all_matches

def save_results_txt(matches, filename="matrix_matches_4x4.txt", start_time=None):
    """Save results in a human-readable text format with statistics"""
    verified_matches = []
    for match in matches:
        A = np.array(match['A'])
        B = np.array(match['B'])
        C = np.array(match['C'])
        if verify_match(A, B, C):
            verified_matches.append((
                tuple(map(tuple, A)),
                tuple(map(tuple, B)),
                tuple(map(tuple, C))
            ))
    
    verified_matches = list(set(verified_matches))  # Deduplicate
    verified_matches = [{
        'A': np.array(m[0]),
        'B': np.array(m[1]),
        'C': np.array(m[2])
    } for m in verified_matches]
    
    total_combinations = get_total_combinations(4)  # Changed to 4
    total_permutations = total_combinations * total_combinations
    execution_time = datetime.now() - start_time if start_time else None
    
    with open(filename, 'w') as f:
        # Write statistics header
        f.write("MATRIX CONCATENATION SEARCH STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"4x4 Matrices:\n")  # Changed to 4x4
        f.write(f"Total permutations tested: {total_permutations:,}\n")
        f.write(f"Matching pairs found: {len(verified_matches)}\n")
        f.write(f"Match rate: {(len(verified_matches)/total_permutations)*100:.8f}%\n")
        if execution_time:
            f.write(f"Execution time: {execution_time}\n")
        f.write("\n" + "=" * 50 + "\n\n")
        
        # Write matches
        f.write(f"Found {len(verified_matches)} matching matrix pairs\n")
        f.write("-" * 50 + "\n\n")
        
        for idx, match in enumerate(verified_matches, 1):
            A, B, C = match['A'], match['B'], match['C']
            f.write(f"Match {idx}:\n")
            # Modified for 4x4 format
            for i in range(4):
                f.write(f"[{' '.join(f'{A[i,j]:2d}' for j in range(4))}] * ")
                f.write(f"[{' '.join(f'{B[i,j]:2d}' for j in range(4))}] = ")
                f.write(f"[{' '.join(f'{C[i,j]:3d}' for j in range(4))}]\n")
            f.write("\n")
            
            f.write("Verification:\n")
            for i in range(4):
                for j in range(4):
                    f.write(f"C[{i},{j}] = {C[i,j]} is concatenation of {A[i,j]} and {B[i,j]}\n")
            f.write("\n" + "-" * 50 + "\n\n")

def get_total_combinations(size):
    """Calculate total possible combinations for 4x4 matrices"""
    return (VALUE_RANGE[1] - VALUE_RANGE[0] + 1) ** (size * size)

def get_device_info():
    return "GPU: CuPy"  # Temporary placeholder

def verify_match(A, B, C):
    """Verify both multiplication and concatenation for 4x4 matrices"""
    # Verify matrix multiplication
    expected_C = np.matmul(A, B)
    if not np.array_equal(expected_C, C):
        return False
    # Verify concatenation property
    for i in range(4):
        for j in range(4):
            concat = (A[i,j] * 10) + B[i,j]
            if C[i,j] != concat:
                return False
    return True

def analyze_results(matches):
    """Analyze results maintaining exact format of base solution"""
    if not matches:
        print("No matches found")
        return
        
    unique_matches = set(tuple(map(tuple, match['A'])) + 
                        tuple(map(tuple, match['B'])) + 
                        tuple(map(tuple, match['C'])) 
                        for match in matches)
    
    print(f"\nFound {len(unique_matches)} unique matching matrix pairs")
    print(f"Total matches including duplicates: {len(matches)}")
    
    # Show only first match
    print("\nFirst match found:")
    match = matches[0]
    print("Matrix A:")
    print(match["A"])
    print("\nMatrix B:")
    print(match["B"])
    print("\nMatrix C (result):")
    print(match["C"])
    
    return len(unique_matches)

if __name__ == "__main__":
    print("Starting matrix search...")
    start_time = datetime.now()
    matches = find_special_matrices_exhaustive()
    execution_time = datetime.now() - start_time
    print(f"\nExecution time: {execution_time}")
    analyze_results(matches)
    save_results_txt(matches, start_time=start_time) 