"""
Unit tests for the Neighbors class methods.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from PyCD.core import Neighbors


class TestNeighbors:
    """Test cases for Neighbors class methods."""
    
    @pytest.fixture
    def mock_material(self):
        """Create a mock Material object for testing."""
        material = Mock()
        material.num_element_types = 2
        material.element_types = ['Fe', 'O']
        material.n_elements_per_unit_cell = np.array([2, 3])
        material.total_elements_per_unit_cell = 5
        material.lattice_matrix = np.array([
            [5.038, 0.000, 0.000],
            [-2.519, 4.363, 0.000],
            [0.000, 0.000, 13.772]
        ])
        material.cartesian_unit_cell_coords = np.array([
            [0.0, 0.0, 4.887],
            [0.0, 0.0, 8.885],
            [1.540, 0.000, 3.445],
            [0.0, 1.454, 3.445],
            [3.498, 3.498, 3.445]
        ])
        
        # Mock the generate_sites method
        def mock_generate_sites(element_type_indices, cell_size):
            mock_sites = Mock()
            total_sites = cell_size.prod() * material.total_elements_per_unit_cell
            mock_sites.cell_coordinates = np.random.rand(total_sites, 3) * 10
            mock_sites.quantum_index_list = np.zeros((total_sites, 5), dtype=int)
            mock_sites.system_element_index_list = np.arange(total_sites)
            return mock_sites
        
        material.generate_sites = mock_generate_sites
        return material
    
    @pytest.fixture
    def neighbors_instance(self, mock_material):
        """Create a Neighbors instance for testing."""
        system_size = np.array([2, 2, 1])
        pbc = [1, 1, 1]
        return Neighbors(mock_material, system_size, pbc)
    
    def test_get_system_element_index_basic(self, neighbors_instance):
        """Test get_system_element_index with basic inputs."""
        system_size = np.array([2, 2, 1])
        
        # Test case 1: First element in first unit cell
        quantum_indices = np.array([0, 0, 0, 0, 0])  # [x, y, z, element_type, element]
        result = neighbors_instance.get_system_element_index(system_size, quantum_indices)
        expected = 0  # First element should have index 0
        assert result == expected
        
        # Test case 2: Second element type, first element
        quantum_indices = np.array([0, 0, 0, 1, 0])  # Second element type (O), first element
        result = neighbors_instance.get_system_element_index(system_size, quantum_indices)
        expected = 2  # Should be after the 2 Fe elements
        assert result == expected
        
        # Test case 3: Different unit cell (y direction first, then x)
        quantum_indices = np.array([0, 1, 0, 0, 0])  # Next unit cell in y direction
        result = neighbors_instance.get_system_element_index(system_size, quantum_indices)
        expected = 5  # After one complete unit cell (5 elements)
        assert result == expected
        
        # Test case 4: Different unit cell (x direction)
        quantum_indices = np.array([1, 0, 0, 0, 0])  # Next unit cell in x direction
        result = neighbors_instance.get_system_element_index(system_size, quantum_indices)
        expected = 10  # After two complete unit cells (2*5 elements)
        assert result == expected
    
    def test_get_quantum_indices_basic(self, neighbors_instance):
        """Test get_quantum_indices with basic inputs."""
        system_size = np.array([2, 2, 1])
        
        # Test case 1: First element
        system_element_index = 0
        result = neighbors_instance.get_quantum_indices(system_size, system_element_index)
        expected = np.array([0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)
        
        # Test case 2: Element in second element type
        system_element_index = 2
        result = neighbors_instance.get_quantum_indices(system_size, system_element_index)
        expected = np.array([0, 0, 0, 1, 0])  # Second element type, first element
        np.testing.assert_array_equal(result, expected)
        
        # Test case 3: Element in next unit cell (y direction)
        system_element_index = 5
        result = neighbors_instance.get_quantum_indices(system_size, system_element_index)
        expected = np.array([0, 1, 0, 0, 0])  # Next unit cell in y direction, first element
        np.testing.assert_array_equal(result, expected)
        
        # Test case 4: Element in next unit cell (x direction)
        system_element_index = 10
        result = neighbors_instance.get_quantum_indices(system_size, system_element_index)
        expected = np.array([1, 0, 0, 0, 0])  # Next unit cell in x direction, first element
        np.testing.assert_array_equal(result, expected)
    
    def test_round_trip_conversion(self, neighbors_instance):
        """Test that get_system_element_index and get_quantum_indices are inverse operations."""
        system_size = np.array([2, 2, 1])
        total_elements = system_size.prod() * neighbors_instance.material.total_elements_per_unit_cell
        
        # Test round trip: quantum_indices -> system_element_index -> quantum_indices
        test_quantum_indices = [
            np.array([0, 0, 0, 0, 0]),  # First element
            np.array([0, 0, 0, 0, 1]),  # Second element of same type
            np.array([0, 0, 0, 1, 0]),  # First element of second type
            np.array([0, 0, 0, 1, 2]),  # Third element of second type
            np.array([1, 0, 0, 0, 0]),  # First element of next unit cell
            np.array([1, 1, 0, 1, 1]),  # Complex case
        ]
        
        for original_quantum_indices in test_quantum_indices:
            # Forward conversion
            system_element_index = neighbors_instance.get_system_element_index(
                system_size, original_quantum_indices)
            
            # Backward conversion
            recovered_quantum_indices = neighbors_instance.get_quantum_indices(
                system_size, system_element_index)
            
            # Verify they match
            np.testing.assert_array_equal(
                original_quantum_indices, recovered_quantum_indices,
                f"Round trip failed for quantum_indices: {original_quantum_indices}")
        
        # Test round trip: system_element_index -> quantum_indices -> system_element_index
        test_system_indices = [0, 1, 2, 4, 5, 7, 10, 15]
        
        for original_system_index in test_system_indices:
            if original_system_index < total_elements:
                # Forward conversion
                quantum_indices = neighbors_instance.get_quantum_indices(
                    system_size, original_system_index)
                
                # Backward conversion
                recovered_system_index = neighbors_instance.get_system_element_index(
                    system_size, quantum_indices)
                
                # Verify they match
                assert original_system_index == recovered_system_index, \
                    f"Round trip failed for system_element_index: {original_system_index}"
    
    def test_different_system_sizes(self, neighbors_instance):
        """Test functions with different system sizes."""
        system_sizes = [
            np.array([1, 1, 1]),
            np.array([2, 1, 1]),
            np.array([2, 2, 1]),
            np.array([3, 2, 2]),
        ]
        
        for system_size in system_sizes:
            total_elements = system_size.prod() * neighbors_instance.material.total_elements_per_unit_cell
            
            # Test several indices for each system size
            for i in range(min(5, total_elements)):
                # Test round trip conversion
                quantum_indices = neighbors_instance.get_quantum_indices(system_size, i)
                recovered_index = neighbors_instance.get_system_element_index(system_size, quantum_indices)
                assert i == recovered_index, \
                    f"Round trip failed for system_size: {system_size}, index: {i}"
    
    def test_edge_cases(self, neighbors_instance):
        """Test edge cases and boundary conditions."""
        system_size = np.array([2, 2, 1])
        
        # Test last element in system
        last_index = system_size.prod() * neighbors_instance.material.total_elements_per_unit_cell - 1
        quantum_indices = neighbors_instance.get_quantum_indices(system_size, last_index)
        recovered_index = neighbors_instance.get_system_element_index(system_size, quantum_indices)
        assert last_index == recovered_index
        
        # Test quantum indices at system boundaries
        max_unit_cell = system_size - 1
        boundary_quantum_indices = np.array([
            max_unit_cell[0], max_unit_cell[1], max_unit_cell[2], 
            neighbors_instance.material.num_element_types - 1,
            neighbors_instance.material.n_elements_per_unit_cell[-1] - 1
        ])
        
        system_index = neighbors_instance.get_system_element_index(system_size, boundary_quantum_indices)
        recovered_quantum_indices = neighbors_instance.get_quantum_indices(system_size, system_index)
        np.testing.assert_array_equal(boundary_quantum_indices, recovered_quantum_indices)
    
    def test_input_validation_concepts(self, neighbors_instance):
        """Test that the functions handle expected input ranges correctly."""
        system_size = np.array([2, 2, 1])
        
        # Test with valid quantum indices
        valid_quantum_indices = np.array([0, 0, 0, 0, 0])
        result = neighbors_instance.get_system_element_index(system_size, valid_quantum_indices)
        assert isinstance(result, (int, np.integer))
        assert result >= 0
        
        # Test with valid system element index
        valid_system_index = 0
        result = neighbors_instance.get_quantum_indices(system_size, valid_system_index)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        assert all(idx >= 0 for idx in result)
    
    def test_consistency_across_element_types(self, neighbors_instance):
        """Test consistency when working with different element types."""
        system_size = np.array([2, 1, 1])
        
        # Test each element type
        for element_type_index in range(neighbors_instance.material.num_element_types):
            for element_index in range(neighbors_instance.material.n_elements_per_unit_cell[element_type_index]):
                quantum_indices = np.array([0, 0, 0, element_type_index, element_index])
                
                system_index = neighbors_instance.get_system_element_index(system_size, quantum_indices)
                recovered_quantum_indices = neighbors_instance.get_quantum_indices(system_size, system_index)
                
                np.testing.assert_array_equal(quantum_indices, recovered_quantum_indices)
    
    def test_mathematical_properties(self, neighbors_instance):
        """Test mathematical properties of the conversion functions."""
        system_size = np.array([2, 2, 1])
        
        # Test that system indices are sequential and unique
        quantum_indices_list = []
        system_indices_list = []
        
        # Generate test cases for each unit cell and element
        for x in range(system_size[0]):
            for y in range(system_size[1]):
                for z in range(system_size[2]):
                    for elem_type in range(neighbors_instance.material.num_element_types):
                        for elem_idx in range(neighbors_instance.material.n_elements_per_unit_cell[elem_type]):
                            quantum_indices = np.array([x, y, z, elem_type, elem_idx])
                            system_index = neighbors_instance.get_system_element_index(system_size, quantum_indices)
                            
                            quantum_indices_list.append(quantum_indices)
                            system_indices_list.append(system_index)
        
        # Check that all system indices are unique
        assert len(set(system_indices_list)) == len(system_indices_list), \
            "System indices should be unique"
        
        # Check that system indices form a continuous sequence starting from 0
        expected_indices = list(range(len(system_indices_list)))
        assert sorted(system_indices_list) == expected_indices, \
            "System indices should form a continuous sequence"