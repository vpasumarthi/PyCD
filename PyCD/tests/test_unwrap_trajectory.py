"""
Test for the unwrap_trajectory functionality in PyCD.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import pytest

from PyCD.core import unwrap_trajectory_file, Analysis
from PyCD import constants


def create_test_trajectory(box_length_ang=10.0, n_steps=20):
    """Create a test trajectory with known boundary crossings"""
    box_length_bohr = box_length_ang * constants.ANG2BOHR
    
    # Create trajectory that crosses boundaries
    positions = np.zeros((n_steps, 3))
    
    # Linear motion in x from 0 to 2.5 box lengths
    x_positions_ang = np.linspace(0, 2.5 * box_length_ang, n_steps)
    x_positions_bohr = x_positions_ang * constants.ANG2BOHR
    
    # Unwrapped positions (ground truth)
    unwrapped_positions = np.copy(positions)
    unwrapped_positions[:, 0] = x_positions_bohr
    
    # Wrapped positions (input for unwrapping)
    wrapped_positions = np.copy(unwrapped_positions)
    wrapped_positions[:, 0] = wrapped_positions[:, 0] % box_length_bohr
    
    return wrapped_positions, unwrapped_positions, box_length_bohr


def test_basic_unwrapping():
    """Test basic unwrapping functionality"""
    box_length_ang = 10.0
    wrapped_traj, true_traj, box_length_bohr = create_test_trajectory(box_length_ang)
    
    system_size = np.array([1, 1, 1])
    lattice_matrix = np.eye(3) * box_length_bohr
    pbc = [1, 1, 1]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wrapped_file = tmpdir / "wrapped_traj.npy"
        np.save(wrapped_file, wrapped_traj)
        
        unwrapped_result = unwrap_trajectory_file(wrapped_file, system_size, lattice_matrix, pbc)
        
        # Check if relative displacements match (what matters for physics)
        true_relative = true_traj - true_traj[0]
        result_relative = unwrapped_result - unwrapped_result[0]
        
        assert np.allclose(true_relative, result_relative, atol=1e-10), \
            "Basic unwrapping failed: relative displacements don't match"


def test_analysis_class_method():
    """Test unwrapping using Analysis class method"""
    box_length_ang = 10.0
    wrapped_traj, true_traj, box_length_bohr = create_test_trajectory(box_length_ang)
    
    system_size = np.array([1, 1, 1])
    lattice_matrix = np.eye(3) * box_length_bohr
    pbc = [1, 1, 1]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wrapped_file = tmpdir / "wrapped_traj.npy"
        np.save(wrapped_file, wrapped_traj)
        
        # Test standalone function
        unwrapped_standalone = unwrap_trajectory_file(wrapped_file, system_size, lattice_matrix, pbc)
        
        # Test Analysis class method
        analysis = Analysis.__new__(Analysis)  # Create without __init__
        unwrapped_analysis = analysis.unwrap_trajectory(wrapped_file, system_size, lattice_matrix, pbc)
        
        assert np.allclose(unwrapped_standalone, unwrapped_analysis, atol=1e-15), \
            "Analysis method and standalone function give different results"


def test_1d_trajectory_format():
    """Test unwrapping 1D trajectory format"""
    box_length_ang = 10.0
    wrapped_traj, true_traj, box_length_bohr = create_test_trajectory(box_length_ang)
    
    system_size = np.array([1, 1, 1])
    lattice_matrix = np.eye(3) * box_length_bohr
    pbc = [1, 1, 1]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test 1D format
        wrapped_1d = wrapped_traj.flatten()
        wrapped_1d_file = tmpdir / "wrapped_1d.npy"
        np.save(wrapped_1d_file, wrapped_1d)
        
        unwrapped_1d = unwrap_trajectory_file(wrapped_1d_file, system_size, lattice_matrix, pbc)
        
        # Should return 1D array
        assert unwrapped_1d.ndim == 1, f"Expected 1D output, got {unwrapped_1d.ndim}D"
        assert len(unwrapped_1d) == len(wrapped_1d), "1D output length doesn't match input"
        
        # Convert back to 2D for comparison with ground truth
        unwrapped_1d_reshaped = unwrapped_1d.reshape(-1, 3)
        result_relative_1d = unwrapped_1d_reshaped - unwrapped_1d_reshaped[0]
        true_relative = true_traj - true_traj[0]
        
        assert np.allclose(true_relative, result_relative_1d, atol=1e-10), \
            "1D unwrapping gives different result than expected"


def test_multiple_species():
    """Test unwrapping trajectory with multiple species"""
    box_length_ang = 10.0
    wrapped_traj, true_traj, box_length_bohr = create_test_trajectory(box_length_ang)
    
    system_size = np.array([1, 1, 1])
    lattice_matrix = np.eye(3) * box_length_bohr
    pbc = [1, 1, 1]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        n_species = 2
        multi_wrapped = np.zeros((len(wrapped_traj), n_species * 3))
        multi_true = np.zeros((len(true_traj), n_species * 3))
        
        for i in range(n_species):
            start_idx = i * 3
            end_idx = start_idx + 3
            multi_wrapped[:, start_idx:end_idx] = wrapped_traj
            multi_true[:, start_idx:end_idx] = true_traj
        
        multi_wrapped_file = tmpdir / "multi_wrapped.npy"
        np.save(multi_wrapped_file, multi_wrapped)
        
        multi_unwrapped = unwrap_trajectory_file(multi_wrapped_file, system_size, lattice_matrix, pbc)
        
        assert multi_unwrapped.shape == multi_wrapped.shape, \
            "Multi-species output shape doesn't match input"
        
        # Check each species separately
        true_relative = true_traj - true_traj[0]
        for i in range(n_species):
            start_idx = i * 3
            end_idx = start_idx + 3
            species_result = multi_unwrapped[:, start_idx:end_idx]
            species_relative = species_result - species_result[0]
            
            assert np.allclose(true_relative, species_relative, atol=1e-10), \
                f"Multi-species unwrapping failed for species {i}"


def test_pbc_options():
    """Test different PBC configurations"""
    box_length_ang = 10.0
    wrapped_traj, true_traj, box_length_bohr = create_test_trajectory(box_length_ang)
    
    system_size = np.array([1, 1, 1])
    lattice_matrix = np.eye(3) * box_length_bohr
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wrapped_file = tmpdir / "wrapped_traj.npy"
        np.save(wrapped_file, wrapped_traj)
        
        # No PBC (should not unwrap)
        unwrapped_no_pbc = unwrap_trajectory_file(wrapped_file, system_size, lattice_matrix, 
                                                 pbc=[0, 0, 0])
        
        assert np.allclose(unwrapped_no_pbc, wrapped_traj, atol=1e-15), \
            "With PBC disabled, output should equal input"
        
        # Partial PBC (only x-direction)
        unwrapped_partial_pbc = unwrap_trajectory_file(wrapped_file, system_size, lattice_matrix, 
                                                      pbc=[1, 0, 0])
        
        partial_relative = unwrapped_partial_pbc - unwrapped_partial_pbc[0]
        true_relative = true_traj - true_traj[0]
        
        assert np.allclose(partial_relative[:, 0], true_relative[:, 0], atol=1e-10), \
            "Partial PBC failed: x-direction not unwrapped"
        assert np.allclose(partial_relative[:, 1:], true_relative[:, 1:], atol=1e-10), \
            "Partial PBC failed: y,z directions should remain unchanged"


def test_save_functionality():
    """Test saving unwrapped trajectory to file"""
    box_length_ang = 10.0
    wrapped_traj, true_traj, box_length_bohr = create_test_trajectory(box_length_ang)
    
    system_size = np.array([1, 1, 1])
    lattice_matrix = np.eye(3) * box_length_bohr
    pbc = [1, 1, 1]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wrapped_file = tmpdir / "wrapped_traj.npy"
        output_file = tmpdir / "unwrapped_output.npy"
        np.save(wrapped_file, wrapped_traj)
        
        unwrapped_result = unwrap_trajectory_file(wrapped_file, system_size, lattice_matrix, 
                                                 pbc, output_file_path=output_file)
        
        assert output_file.exists(), "Output file was not created"
        
        saved_traj = np.load(output_file)
        assert np.allclose(unwrapped_result, saved_traj, atol=1e-15), \
            "Saved trajectory doesn't match returned trajectory"


def test_error_handling():
    """Test error handling for invalid inputs"""
    system_size = np.array([1, 1, 1])
    lattice_matrix = np.eye(3) * 10.0
    
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        unwrap_trajectory_file("nonexistent.npy", system_size, lattice_matrix)
    
    # Create valid file for other tests
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wrapped_file = tmpdir / "wrapped_traj.npy"
        test_traj = np.zeros((10, 3))
        np.save(wrapped_file, test_traj)
        
        # Invalid system size
        with pytest.raises(ValueError):
            unwrap_trajectory_file(wrapped_file, [1, 1], lattice_matrix)  # Wrong length
        
        # Invalid lattice matrix
        with pytest.raises(ValueError):
            unwrap_trajectory_file(wrapped_file, system_size, np.eye(2))  # Wrong shape
        
        # Invalid PBC
        with pytest.raises(ValueError):
            unwrap_trajectory_file(wrapped_file, system_size, lattice_matrix, pbc=[1, 1])  # Wrong length