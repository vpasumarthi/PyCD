"""Test HDF5 trajectory I/O functionality."""

import tempfile
import numpy as np
import pytest
from pathlib import Path

try:
    import h5py
    from PyCD.hdf5_io import HDF5TrajectoryWriter, save_trajectory_hdf5
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_hdf5_trajectory_writer():
    """Test basic HDF5 trajectory writing functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        hdf5_file = tmpdir / "test_trajectory.h5"
        
        n_atoms = 3
        n_frames = 5
        
        # Create test data
        coordinates = np.random.rand(n_frames, n_atoms, 3) * 10.0  # Random coordinates in Bohr
        time_data = np.arange(n_frames, dtype=float)
        
        # Write HDF5 file
        with HDF5TrajectoryWriter(hdf5_file, n_atoms, 1) as writer:
            writer.write_frames(coordinates, time_data)
        
        # Verify file was created
        assert hdf5_file.exists(), "HDF5 file was not created"
        
        # Read back and verify data
        with h5py.File(hdf5_file, 'r') as f:
            assert 'coordinates' in f, "Coordinates dataset not found"
            assert 'time' in f, "Time dataset not found"
            assert 'topology' in f, "Topology group not found"
            
            # Check dimensions
            coords_shape = f['coordinates'].shape
            time_shape = f['time'].shape
            
            assert coords_shape == (n_frames, n_atoms, 3), f"Wrong coordinates shape: {coords_shape}"
            assert time_shape == (n_frames,), f"Wrong time shape: {time_shape}"
            
            # Check units attributes
            coords_units = f['coordinates'].attrs['units']
            time_units = f['time'].attrs['units']
            assert coords_units == 'nanometers' or coords_units == b'nanometers'
            assert time_units == 'picoseconds' or time_units == b'picoseconds'
            
            # Verify topology information
            assert 'topology/atoms' in f, "Atoms group not found in topology"
            assert 'topology/atoms/name' in f, "Atom names not found"
            assert 'topology/atoms/element' in f, "Atom elements not found"
            assert 'topology/atoms/index' in f, "Atom indices not found"
            
            # Check atom count
            assert len(f['topology/atoms/name']) == n_atoms, "Wrong number of atoms in topology"


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_hdf5_single_frame():
    """Test writing a single frame to HDF5."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        hdf5_file = tmpdir / "single_frame.h5"
        
        n_atoms = 2
        coordinates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 atoms
        time_value = 0.0
        
        with HDF5TrajectoryWriter(hdf5_file, n_atoms, 1) as writer:
            writer.write_frame(coordinates, time_value)
        
        with h5py.File(hdf5_file, 'r') as f:
            assert f['coordinates'].shape == (1, n_atoms, 3)
            assert f['time'].shape == (1,)


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_hdf5_convenience_function():
    """Test the convenience save_trajectory_hdf5 function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        hdf5_file = tmpdir / "convenience_test.h5"
        
        # Test with 2D coordinates
        coordinates = np.random.rand(4, 6)  # 4 frames, 2 atoms * 3 coords
        time_data = np.arange(4, dtype=float)
        
        save_trajectory_hdf5(hdf5_file, coordinates, time_data, n_atoms=2)
        
        assert hdf5_file.exists()
        
        with h5py.File(hdf5_file, 'r') as f:
            assert f['coordinates'].shape == (4, 2, 3)
            assert f['time'].shape == (4,)


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_hdf5_1d_coordinates():
    """Test HDF5 writing with 1D coordinate input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        hdf5_file = tmpdir / "1d_coords.h5"
        
        # 1D coordinates: single frame, 2 atoms
        coordinates = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        save_trajectory_hdf5(hdf5_file, coordinates)
        
        with h5py.File(hdf5_file, 'r') as f:
            assert f['coordinates'].shape == (1, 2, 3)
            assert f['time'].shape == (1,)


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_hdf5_cross_format_consistency():
    """Test consistency between numpy and HDF5 formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        npy_file = tmpdir / "test_traj.npy"
        hdf5_file = tmpdir / "test_traj.h5"
        
        # Create test trajectory data
        n_frames = 3
        n_atoms = 2
        coordinates = np.random.rand(n_frames, n_atoms * 3) * 10.0
        time_data = np.arange(n_frames, dtype=float)
        
        # Save in numpy format
        np.save(npy_file, coordinates)
        
        # Save in HDF5 format
        save_trajectory_hdf5(hdf5_file, coordinates, time_data, n_atoms=n_atoms)
        
        # Load both and compare (after unit conversion)
        npy_data = np.load(npy_file)
        
        with h5py.File(hdf5_file, 'r') as f:
            hdf5_coords = f['coordinates'][:]
            
        # Reshape HDF5 data to match numpy format
        hdf5_coords_flat = hdf5_coords.reshape(n_frames, n_atoms * 3)
        
        # Convert HDF5 coordinates back to Bohr for comparison
        # HDF5 stores in nm, numpy in Bohr
        from PyCD import constants
        hdf5_coords_bohr = hdf5_coords_flat / (0.1 / constants.ANG2BOHR)
        
        # Compare (allowing for floating point precision)
        np.testing.assert_allclose(npy_data, hdf5_coords_bohr, rtol=1e-6)


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_hdf5_error_handling():
    """Test HDF5 error handling."""
    # Test with invalid file path
    invalid_path = "/invalid/path/test.h5"
    
    with pytest.raises(Exception):
        with HDF5TrajectoryWriter(invalid_path, 2, 1) as writer:
            pass


if __name__ == "__main__":
    pytest.main([__file__])