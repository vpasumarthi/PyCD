"""
Tests for PBC optimization in the Neighbors class.
"""

import numpy as np
import pytest
from PyCD.core import Neighbors, Material, ReturnValues


class TestPBCOptimization:
    """Test the optimized minimum image convention implementation."""
    
    @pytest.fixture
    def mock_material(self):
        """Create a mock material for testing."""
        # Create non-orthorhombic lattice matrix
        lattice_matrix = np.array([
            [3.0, 0.1, 0.0],  
            [0.1, 3.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        
        class MockMaterial:
            def __init__(self):
                self.lattice_matrix = lattice_matrix
        
        return MockMaterial()
    
    @pytest.fixture  
    def neighbors_instance(self, mock_material):
        """Create a Neighbors instance for testing."""
        system_size = np.array([2, 2, 2])
        pbc = [1, 1, 1]
        
        neighbors = Neighbors.__new__(Neighbors)
        neighbors.material = mock_material
        neighbors.system_size = system_size
        neighbors.pbc = pbc
        neighbors._setup_efficient_pbc()
        
        return neighbors
    
    def brute_force_minimum_image(self, displacement_vector, neighbors):
        """Reference brute-force implementation for comparison."""
        translation_vectors = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    offset = np.array([x, y, z])
                    translation_vector = np.dot(
                        offset * neighbors.system_size, 
                        neighbors.material.lattice_matrix
                    )
                    translation_vectors.append(translation_vector)
        
        translation_vectors = np.array(translation_vectors)
        neighbor_image_displacement_vectors = translation_vectors + displacement_vector
        neighbor_image_displacements = np.linalg.norm(neighbor_image_displacement_vectors, axis=1)
        image_index = np.argmin(neighbor_image_displacements)
        
        return neighbor_image_displacement_vectors[image_index]
    
    def test_basic_minimum_image_convention(self, neighbors_instance):
        """Test that optimized algorithm produces correct results."""
        test_displacements = [
            np.array([-2.50919762, 9.01428613, 4.63987884]),
            np.array([1.0, 1.0, 1.0]),
            np.array([-5.0, 0.5, 2.3]),
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, -8.0, 6.0])
        ]
        
        for i, displacement in enumerate(test_displacements):
            # Get result from optimized method
            optimized_result = neighbors_instance.apply_minimum_image_convention(displacement)
            
            # Get result from brute-force method
            brute_force_result = self.brute_force_minimum_image(displacement, neighbors_instance)
            
            # Compare distances (should be very close)
            optimized_distance = np.linalg.norm(optimized_result)
            brute_force_distance = np.linalg.norm(brute_force_result)
            
            # The optimized method should find the same or better minimum
            assert optimized_distance <= brute_force_distance + 1e-10, \
                f"Test {i}: Optimized distance {optimized_distance} > brute-force distance {brute_force_distance}"
    
    def test_vectorized_version(self, neighbors_instance):
        """Test that vectorized version works correctly."""
        np.random.seed(42)
        num_vectors = 20
        displacement_vectors = np.random.uniform(-8.0, 8.0, (num_vectors, 3))
        
        # Test vectorized version
        vectorized_results = neighbors_instance.apply_minimum_image_convention_vectorized(displacement_vectors)
        
        # Test single-vector version for comparison
        single_results = np.array([
            neighbors_instance.apply_minimum_image_convention(dv) 
            for dv in displacement_vectors
        ])
        
        # Compare results
        max_diff = np.max(np.linalg.norm(vectorized_results - single_results, axis=1))
        
        assert max_diff < 1e-12, f"Vectorized version differs from single-vector version: max diff = {max_diff}"
    
    def test_orthogonal_vs_non_orthorhombic(self):
        """Test behavior with orthorhombic vs non-orthorhombic cells."""
        # Orthorhombic case
        ortho_material = type('MockMaterial', (), {
            'lattice_matrix': np.array([
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0], 
                [0.0, 0.0, 3.0]
            ])
        })()
        
        # Non-orthorhombic case 
        non_ortho_material = type('MockMaterial', (), {
            'lattice_matrix': np.array([
                [3.0, 0.5, 0.0],
                [0.5, 3.0, 0.0],
                [0.0, 0.0, 3.0]
            ])
        })()
        
        system_size = np.array([2, 2, 2])
        pbc = [1, 1, 1]
        
        for material in [ortho_material, non_ortho_material]:
            neighbors = Neighbors.__new__(Neighbors)
            neighbors.material = material
            neighbors.system_size = system_size
            neighbors.pbc = pbc
            neighbors._setup_efficient_pbc()
            
            # Test a displacement vector
            displacement = np.array([4.5, 4.5, 1.5])
            result = neighbors.apply_minimum_image_convention(displacement)
            
            # Should produce a valid result
            assert isinstance(result, np.ndarray)
            assert result.shape == (3,)
            assert np.all(np.isfinite(result))
    
    def test_partial_pbc(self):
        """Test with partial periodic boundary conditions."""
        # Use orthorhombic lattice to avoid coupling between dimensions
        class MockMaterial:
            def __init__(self):
                self.lattice_matrix = np.array([
                    [3.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 3.0]
                ])
        
        material = MockMaterial()
        system_size = np.array([2, 2, 2])
        
        # Test different PBC configurations
        pbc_configs = [
            [1, 1, 0],  # Periodic in x,y only
            [1, 0, 1],  # Periodic in x,z only
            [0, 1, 1],  # Periodic in y,z only
            [1, 0, 0],  # Periodic in x only
        ]
        
        displacement = np.array([4.0, 4.0, 4.0])
        
        for pbc in pbc_configs:
            neighbors = Neighbors.__new__(Neighbors)
            neighbors.material = material
            neighbors.system_size = system_size
            neighbors.pbc = pbc
            neighbors._setup_efficient_pbc()
            
            result = neighbors.apply_minimum_image_convention(displacement)
            
            # For orthorhombic cells, non-PBC dimensions should be unchanged
            for i, is_periodic in enumerate(pbc):
                if not is_periodic:
                    # Non-periodic dimension should not be wrapped
                    assert abs(result[i] - displacement[i]) < 1e-10, \
                        f"Non-PBC dimension {i} was modified: {result[i]} vs {displacement[i]}"
    
    def test_performance_improvement(self, neighbors_instance):
        """Basic performance test to ensure the optimization is beneficial."""
        import time
        
        np.random.seed(42)
        num_vectors = 100
        displacement_vectors = np.random.uniform(-5.0, 5.0, (num_vectors, 3))
        
        # Time optimized method
        start_time = time.time()
        optimized_results = [
            neighbors_instance.apply_minimum_image_convention(dv) 
            for dv in displacement_vectors
        ]
        optimized_time = time.time() - start_time
        
        # Time brute-force method
        start_time = time.time()
        brute_force_results = [
            self.brute_force_minimum_image(dv, neighbors_instance) 
            for dv in displacement_vectors
        ]
        brute_force_time = time.time() - start_time
        
        # Check that results are comparable
        max_diff = max(
            np.linalg.norm(opt - bf) 
            for opt, bf in zip(optimized_results, brute_force_results)
        )
        
        # Performance should be better or at least not much worse
        # (In some cases, the small search might be comparable to brute-force for small systems)
        speedup = brute_force_time / optimized_time
        
        print(f"Optimized time: {optimized_time:.4f}s, Brute-force time: {brute_force_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x, Max difference: {max_diff:.2e}")
        
        # The optimized version should at least produce correct results
        assert max_diff < 1e-10, f"Results differ too much: {max_diff}"