#!/usr/bin/env python3
"""
Debug the PBC implementation
"""

import numpy as np

def debug_pbc_implementation():
    """Debug the PBC implementation step by step"""
    
    # Define a simple test case
    lattice_matrix = np.array([
        [3.0, 0.1, 0.0],
        [0.1, 3.0, 0.0],
        [0.0, 0.0, 3.0]
    ])
    
    system_size = np.array([2, 2, 2])
    pbc = np.array([1, 1, 1], dtype=bool)
    
    # Test displacement vector (from the failing case)
    displacement_vector = np.array([-2.50919762, 9.01428613, 4.63987884])
    
    print("=== Debugging PBC Implementation ===")
    print(f"Lattice matrix:\n{lattice_matrix}")
    print(f"System size: {system_size}")
    print(f"PBC: {pbc}")
    print(f"Displacement vector: {displacement_vector}")
    
    # Method 1: Efficient approach
    print("\n--- Efficient Method ---")
    simulation_cell_matrix = lattice_matrix * system_size[:, np.newaxis]
    print(f"Simulation cell matrix:\n{simulation_cell_matrix}")
    
    simulation_cell_matrix_inv = np.linalg.inv(simulation_cell_matrix)
    print(f"Inverse simulation cell matrix:\n{simulation_cell_matrix_inv}")
    
    # Convert to fractional coordinates
    frac_displacement = np.dot(displacement_vector, simulation_cell_matrix_inv)
    print(f"Fractional displacement: {frac_displacement}")
    
    # Apply wrapping
    frac_displacement_wrapped = frac_displacement.copy()
    frac_displacement_wrapped[pbc] = frac_displacement_wrapped[pbc] - np.floor(frac_displacement_wrapped[pbc] + 0.5)
    print(f"Wrapped fractional displacement: {frac_displacement_wrapped}")
    
    # Let's also check what offset this corresponds to
    offset_from_efficient = -np.floor(frac_displacement + 0.5)
    print(f"Offset from efficient method: {offset_from_efficient}")
    
    # Convert back to Cartesian
    efficient_result = np.dot(frac_displacement_wrapped, simulation_cell_matrix)
    print(f"Efficient result: {efficient_result}")
    
    # Let's verify: displacement + translation_vector should equal efficient_result
    expected_translation = np.dot(offset_from_efficient * system_size, lattice_matrix)
    print(f"Expected translation vector: {expected_translation}")
    reconstructed_result = displacement_vector + expected_translation
    print(f"Reconstructed result: {reconstructed_result}")
    print(f"Difference (efficient vs reconstructed): {np.linalg.norm(efficient_result - reconstructed_result)}")
    
    # Check what offset the brute-force method found
    brute_force_translation = np.array([-0.2, -6., -6.])
    # To find the offset, we solve: offset * system_size @ lattice_matrix = translation
    # This gives: offset * system_size = translation @ lattice_matrix^(-1)
    lattice_inv = np.linalg.inv(lattice_matrix)
    brute_force_offset_scaled = brute_force_translation @ lattice_inv
    brute_force_offset = brute_force_offset_scaled / system_size
    print(f"Brute-force offset: {brute_force_offset}")
    print(f"Brute-force offset (rounded): {np.round(brute_force_offset)}")
    
    # What fractional coordinate does this correspond to?
    brute_force_frac_wrapped = frac_displacement + brute_force_offset
    print(f"Brute-force wrapped fractional: {brute_force_frac_wrapped}")
    
    # Method 2: Brute-force approach
    print("\n--- Brute-force Method ---")
    translation_vectors = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                offset = np.array([x, y, z])
                # This is how the original code constructs translation vectors
                translation_vector = np.dot(offset * system_size, lattice_matrix)
                translation_vectors.append(translation_vector)
                print(f"Offset {offset}: translation vector {translation_vector}")
    
    translation_vectors = np.array(translation_vectors)
    
    # Find minimum image
    neighbor_image_displacement_vectors = translation_vectors + displacement_vector
    neighbor_image_displacements = np.linalg.norm(neighbor_image_displacement_vectors, axis=1)
    
    print(f"\nAll image displacement vectors:")
    for i, (trans, disp_vec, dist) in enumerate(zip(translation_vectors, neighbor_image_displacement_vectors, neighbor_image_displacements)):
        print(f"  {i}: trans={trans}, disp_vec={disp_vec}, dist={dist:.6f}")
    
    image_index = np.argmin(neighbor_image_displacements)
    brute_force_result = neighbor_image_displacement_vectors[image_index]
    
    print(f"\nMinimum image index: {image_index}")
    print(f"Brute-force result: {brute_force_result}")
    
    # Compare
    diff = np.linalg.norm(efficient_result - brute_force_result)
    print(f"\nDifference: {diff}")
    
    # Let's also check what the simulation cell vectors actually are
    print(f"\n--- Simulation Cell Analysis ---")
    print(f"Simulation cell vector 0: {simulation_cell_matrix[0, :]}")
    print(f"Simulation cell vector 1: {simulation_cell_matrix[1, :]}")
    print(f"Simulation cell vector 2: {simulation_cell_matrix[2, :]}")
    
    # Compare with how translation vectors are constructed
    print(f"\n--- Translation Vector Analysis ---")
    for i, offset in enumerate([[1,0,0], [0,1,0], [0,0,1]]):
        offset = np.array(offset)
        manual_trans = np.dot(offset * system_size, lattice_matrix)
        sim_cell_vec = simulation_cell_matrix[i, :]
        print(f"Offset {offset}: manual={manual_trans}, sim_cell={sim_cell_vec}, diff={np.linalg.norm(manual_trans - sim_cell_vec)}")

if __name__ == "__main__":
    debug_pbc_implementation()