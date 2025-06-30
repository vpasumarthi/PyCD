#!/usr/bin/env python3
"""
Simple test for the PBC optimization without Material dependencies
"""

import numpy as np
import sys
sys.path.insert(0, '.')

def test_efficient_minimum_image():
    """Test the efficient minimum image algorithm directly"""
    print("Testing efficient minimum image convention...")
    
    # Define a simple non-orthorhombic simulation cell
    lattice_matrix = np.array([
        [3.0, 0.1, 0.0],  # Non-orthorhombic
        [0.1, 3.0, 0.0],
        [0.0, 0.0, 3.0]
    ])
    
    system_size = np.array([2, 2, 2])
    pbc = np.array([1, 1, 1], dtype=bool)
    
    # Calculate simulation cell matrix
    simulation_cell_matrix = lattice_matrix * system_size[:, np.newaxis]
    simulation_cell_matrix_inv = np.linalg.inv(simulation_cell_matrix)
    
    def efficient_minimum_image(displacement_vector):
        """Efficient minimum image convention"""
        # Convert to fractional coordinates
        frac_displacement = np.dot(displacement_vector, simulation_cell_matrix_inv)
        
        # For non-PBC dimensions, no offset is allowed
        # For PBC dimensions, check offsets of -1, 0, +1 around the wrapped position
        candidate_offsets = []
        
        for dx in ([-1, 0, 1] if pbc[0] else [0]):
            for dy in ([-1, 0, 1] if pbc[1] else [0]):
                for dz in ([-1, 0, 1] if pbc[2] else [0]):
                    candidate_offsets.append([dx, dy, dz])
        
        # Start with the simple wrapped solution as baseline
        base_offset = np.zeros(3)
        base_offset[pbc] = -np.floor(frac_displacement[pbc] + 0.5)
        
        best_distance = float('inf')
        best_displacement = None
        
        # Check each candidate offset
        for offset_delta in candidate_offsets:
            offset = base_offset + np.array(offset_delta)
            
            # Calculate the wrapped fractional displacement
            frac_wrapped = frac_displacement + offset
            
            # Convert back to Cartesian
            cart_displacement = np.dot(frac_wrapped, simulation_cell_matrix)
            
            # Calculate distance
            distance = np.linalg.norm(cart_displacement)
            
            if distance < best_distance:
                best_distance = distance
                best_displacement = cart_displacement
        
        return best_displacement
    
    def brute_force_minimum_image(displacement_vector):
        """Original brute-force approach"""
        # Generate all translation vectors
        translation_vectors = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    offset = np.array([x, y, z])
                    translation_vector = np.dot(offset * system_size, lattice_matrix)
                    translation_vectors.append(translation_vector)
        
        translation_vectors = np.array(translation_vectors)
        
        # Find minimum image
        neighbor_image_displacement_vectors = translation_vectors + displacement_vector
        neighbor_image_displacements = np.linalg.norm(neighbor_image_displacement_vectors, axis=1)
        image_index = np.argmin(neighbor_image_displacements)
        
        return neighbor_image_displacement_vectors[image_index]
    
    # Test with random displacement vectors
    np.random.seed(42)
    num_tests = 100
    
    max_diff = 0.0
    
    for i in range(num_tests):
        # Generate random displacement vector
        displacement_vector = np.random.uniform(-10.0, 10.0, 3)
        
        # Get results from both methods
        efficient_result = efficient_minimum_image(displacement_vector)
        brute_force_result = brute_force_minimum_image(displacement_vector)
        
        # Calculate difference
        diff = np.linalg.norm(efficient_result - brute_force_result)
        max_diff = max(max_diff, diff)
        
        if diff > 1e-10:
            print(f"MISMATCH at test {i}:")
            print(f"  Displacement: {displacement_vector}")
            print(f"  Efficient: {efficient_result}")
            print(f"  Brute-force: {brute_force_result}")
            print(f"  Difference: {diff}")
            return False
    
    print(f"✓ All {num_tests} tests passed!")
    print(f"  Maximum difference: {max_diff:.2e}")
    
    return True

def test_performance():
    """Test performance improvement"""
    print("\nTesting performance...")
    
    # Define simulation cell
    lattice_matrix = np.array([
        [3.0, 0.1, 0.0],
        [0.1, 3.0, 0.0], 
        [0.0, 0.0, 3.0]
    ])
    
    system_size = np.array([3, 3, 3])
    pbc = np.array([1, 1, 1], dtype=bool)
    
    simulation_cell_matrix = lattice_matrix * system_size[:, np.newaxis]
    simulation_cell_matrix_inv = np.linalg.inv(simulation_cell_matrix)
    
    def efficient_minimum_image_vectorized(displacement_vectors):
        """Vectorized efficient version"""
        results = []
        for displacement_vector in displacement_vectors:
            # Convert to fractional coordinates
            frac_displacement = np.dot(displacement_vector, simulation_cell_matrix_inv)
            
            # For non-PBC dimensions, no offset is allowed
            # For PBC dimensions, check offsets of -1, 0, +1 around the wrapped position
            candidate_offsets = []
            
            for dx in ([-1, 0, 1] if pbc[0] else [0]):
                for dy in ([-1, 0, 1] if pbc[1] else [0]):
                    for dz in ([-1, 0, 1] if pbc[2] else [0]):
                        candidate_offsets.append([dx, dy, dz])
            
            # Start with the simple wrapped solution as baseline
            base_offset = np.zeros(3)
            base_offset[pbc] = -np.floor(frac_displacement[pbc] + 0.5)
            
            best_distance = float('inf')
            best_displacement = None
            
            # Check each candidate offset
            for offset_delta in candidate_offsets:
                offset = base_offset + np.array(offset_delta)
                
                # Calculate the wrapped fractional displacement
                frac_wrapped = frac_displacement + offset
                
                # Convert back to Cartesian
                cart_displacement = np.dot(frac_wrapped, simulation_cell_matrix)
                
                # Calculate distance
                distance = np.linalg.norm(cart_displacement)
                
                if distance < best_distance:
                    best_distance = distance
                    best_displacement = cart_displacement
            
            results.append(best_displacement)
        
        return np.array(results)
    
    def brute_force_vectorized(displacement_vectors):
        """Vectorized brute-force version"""
        # Generate all translation vectors
        translation_vectors = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    offset = np.array([x, y, z])
                    translation_vector = np.dot(offset * system_size, lattice_matrix)
                    translation_vectors.append(translation_vector)
        
        translation_vectors = np.array(translation_vectors)
        
        results = []
        for displacement_vector in displacement_vectors:
            neighbor_image_displacement_vectors = translation_vectors + displacement_vector
            neighbor_image_displacements = np.linalg.norm(neighbor_image_displacement_vectors, axis=1)
            image_index = np.argmin(neighbor_image_displacements)
            results.append(neighbor_image_displacement_vectors[image_index])
        
        return np.array(results)
    
    # Generate test data
    np.random.seed(42)
    num_vectors = 1000
    displacement_vectors = np.random.uniform(-10.0, 10.0, (num_vectors, 3))
    
    import time
    
    # Time efficient method
    start_time = time.time()
    efficient_results = efficient_minimum_image_vectorized(displacement_vectors)
    efficient_time = time.time() - start_time
    
    # Time brute-force method
    start_time = time.time()
    brute_force_results = brute_force_vectorized(displacement_vectors)
    brute_force_time = time.time() - start_time
    
    # Check correctness
    max_diff = np.max(np.linalg.norm(efficient_results - brute_force_results, axis=1))
    
    speedup = brute_force_time / efficient_time
    
    print(f"  Efficient method: {efficient_time:.4f} seconds")
    print(f"  Brute-force method: {brute_force_time:.4f} seconds")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Maximum difference: {max_diff:.2e}")
    
    return speedup > 1.0 and max_diff < 1e-10

if __name__ == "__main__":
    success = True
    
    success &= test_efficient_minimum_image()
    success &= test_performance()
    
    if success:
        print("\n✅ All tests passed! The PBC optimization is working correctly.")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)