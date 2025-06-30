#!/usr/bin/env python3
"""
Test script to verify that the optimized minimum image convention 
produces the same results as the original brute-force implementation.
"""

import numpy as np
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, '.')

from PyCD.core import Neighbors, Material, ReturnValues

def create_test_material():
    """Create a simple test material for testing"""
    # Create mock material parameters
    params = {
        'name': 'test',
        'species_types': ['Fe'],
        'species_charge_list': [0],
        'species_to_element_type_map': {'Fe': [0]},
        'class_list': {'Fe': [1]},
        'charge_types': {'test': {'Fe': 0}},
        'vn': 1e13,
        'lambda_values': {},
        'v_ab': {}
    }
    
    # Create mock POSCAR info
    lattice_matrix = np.array([
        [3.0, 0.1, 0.0],  # Non-orthorhombic
        [0.1, 3.0, 0.0],
        [0.0, 0.0, 3.0]
    ])
    
    poscar_info = {
        'lattice_matrix': lattice_matrix,
        'element_types': ['Fe'],
        'num_elements': [1],
        'total_elements': 1,
        'coordinate_type': 'Direct',
        'coordinates': np.array([[0.0, 0.0, 0.0]])
    }
    
    # Mock read_poscar function
    original_read_poscar = None
    if hasattr(sys.modules.get('PyCD.io', None), 'read_poscar'):
        import PyCD.io
        original_read_poscar = PyCD.io.read_poscar
        PyCD.io.read_poscar = lambda x: poscar_info
    
    config_params = ReturnValues(**params)
    config_params.input_coord_file_location = 'dummy'
    
    material = Material(config_params)
    
    # Restore original function
    if original_read_poscar:
        PyCD.io.read_poscar = original_read_poscar
    
    return material

def brute_force_minimum_image(displacement_vector, system_translational_vector_list):
    """Original brute-force implementation for comparison"""
    neighbor_image_displacement_vectors = system_translational_vector_list + displacement_vector
    neighbor_image_displacements = np.linalg.norm(neighbor_image_displacement_vectors, axis=1)
    image_index = np.argmin(neighbor_image_displacements)
    return neighbor_image_displacement_vectors[image_index]

def test_minimum_image_convention():
    """Test that efficient and brute-force methods give same results"""
    print("Testing minimum image convention optimization...")
    
    # Create test material
    material = create_test_material()
    
    # Test different system sizes and PBC configurations
    test_cases = [
        ([2, 2, 2], [1, 1, 1]),  # Full PBC
        ([3, 3, 1], [1, 1, 0]),  # Partial PBC
        ([2, 2, 2], [1, 0, 1]),  # Partial PBC
    ]
    
    for system_size, pbc in test_cases:
        print(f"Testing system_size={system_size}, pbc={pbc}")
        
        # Create neighbors object
        neighbors = Neighbors(material, np.array(system_size), pbc)
        
        # Test random displacement vectors
        np.random.seed(42)  # For reproducible results
        num_tests = 100
        
        for i in range(num_tests):
            # Generate random displacement vector
            displacement_vector = np.random.uniform(-5.0, 5.0, 3)
            
            # Get result from efficient method
            efficient_result = neighbors.apply_minimum_image_convention(displacement_vector)
            
            # Get result from brute-force method
            brute_force_result = brute_force_minimum_image(
                displacement_vector, neighbors.system_translational_vector_list)
            
            # Compare results (should be very close)
            distance_diff = np.linalg.norm(efficient_result - brute_force_result)
            
            if distance_diff > 1e-10:
                print(f"MISMATCH: efficient={efficient_result}, brute_force={brute_force_result}")
                print(f"Displacement vector: {displacement_vector}")
                print(f"Distance difference: {distance_diff}")
                return False
        
        print(f"✓ Passed {num_tests} tests for system_size={system_size}, pbc={pbc}")
    
    return True

def test_vectorized_version():
    """Test the vectorized version against single-vector version"""
    print("Testing vectorized minimum image convention...")
    
    material = create_test_material()
    neighbors = Neighbors(material, np.array([2, 2, 2]), [1, 1, 1])
    
    # Generate test displacement vectors
    np.random.seed(42)
    num_vectors = 50
    displacement_vectors = np.random.uniform(-5.0, 5.0, (num_vectors, 3))
    
    # Test vectorized version
    vectorized_results = neighbors.apply_minimum_image_convention_vectorized(displacement_vectors)
    
    # Test single-vector version for comparison
    single_results = np.array([
        neighbors.apply_minimum_image_convention(dv) for dv in displacement_vectors
    ])
    
    # Compare results
    max_diff = np.max(np.linalg.norm(vectorized_results - single_results, axis=1))
    
    if max_diff > 1e-10:
        print(f"MISMATCH: max difference = {max_diff}")
        return False
    
    print(f"✓ Vectorized version matches single-vector version (max diff: {max_diff:.2e})")
    return True

def test_performance_comparison():
    """Compare performance of old vs new implementation"""
    print("Performance comparison (basic timing)...")
    
    material = create_test_material()
    neighbors = Neighbors(material, np.array([3, 3, 3]), [1, 1, 1])
    
    # Generate test data
    np.random.seed(42)
    num_vectors = 1000
    displacement_vectors = np.random.uniform(-5.0, 5.0, (num_vectors, 3))
    
    import time
    
    # Time efficient method
    start_time = time.time()
    efficient_results = neighbors.apply_minimum_image_convention_vectorized(displacement_vectors)
    efficient_time = time.time() - start_time
    
    # Time brute-force method
    start_time = time.time()
    brute_force_results = np.array([
        brute_force_minimum_image(dv, neighbors.system_translational_vector_list) 
        for dv in displacement_vectors
    ])
    brute_force_time = time.time() - start_time
    
    speedup = brute_force_time / efficient_time
    
    print(f"Efficient method: {efficient_time:.4f} seconds")
    print(f"Brute-force method: {brute_force_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify results are the same
    max_diff = np.max(np.linalg.norm(efficient_results - brute_force_results, axis=1))
    print(f"Maximum difference: {max_diff:.2e}")
    
    return speedup > 1.0 and max_diff < 1e-10

if __name__ == "__main__":
    success = True
    
    try:
        success &= test_minimum_image_convention()
        success &= test_vectorized_version() 
        success &= test_performance_comparison()
        
        if success:
            print("\n✅ All tests passed! The optimization is working correctly.")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)