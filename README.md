# Interesting_Matrix_Finder_2x2_4x4
Find pairs of 2x2 and 4x4 matricies where the dot product is equivalent to result of concatenating each term with the respective term in the same position.

![image](https://github.com/user-attachments/assets/e004d2a5-d8e8-4714-8bad-fb6b0b44cbb2)

## Overview
This project implements a highly optimized search algorithm to find special matrix pairs that satisfy both multiplication and concatenation properties. We've created two versions:
- `matrices.py`: Optimized for 2x2 matrices
- `matrices_4x4.py`: Extended solution for 4x4 matrices

## Problem Statement
We search for matrix pairs (A, B) where:
1. Each element is a single digit (1-9)
2. The matrix multiplication result (C = A × B) equals
3. The element-wise digit concatenation of A and B
   [a b] × [e f] = [ae bf]
   [c d] [g h] [cg dh]
5. Where each position in the result is both:
- The result of matrix multiplication
- The concatenation of corresponding digits
  
## Technical Implementation

### Performance Optimizations
- CUDA GPU acceleration via CuPy
- Parallel processing using ProcessPoolExecutor
- Vectorized operations with NumPy
- Optimized bit operations for matrix generation
- Smart batch processing to maximize GPU utilization
- Memory-efficient data structures

### Key Features
- Unified memory management for GPU operations
- Progress tracking with tqdm
- Comprehensive result verification
- Deduplication of found matches
- Detailed statistics and timing information

### Batch Processing Strategy
- Dynamic batch sizing based on matrix dimensions
- Pre-filtering of invalid matrix combinations
- Efficient GPU memory management
- Vectorized concatenation checks

## Requirements
- Python 3.x
- CUDA-capable GPU
- NumPy
- CuPy
- tqdm

## Output
Results are saved to:
- `matrix_matches.txt` (2x2 results)
- `matrix_matches_4x4.txt` (4x4 results)

Including:
- Total permutations tested
- Number of matches found
- Match rate percentage
- Execution time
- Detailed match verification

## Performance Considerations
- 2x2 search space: 9^8 combinations
- 4x4 search space: 9^32 combinations
- GPU memory optimization crucial for 4x4
- Parallel processing to utilize multiple CPU cores
- Efficient filtering to reduce search space
