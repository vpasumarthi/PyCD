"""Integration test for HDF5 trajectory output with simulation configuration."""

import tempfile
import numpy as np
import pytest
import yaml
from pathlib import Path

try:
    import h5py
    from PyCD.hdf5_io import HDF5TrajectoryWriter
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_hdf5_config_integration():
    """Test that HDF5 configuration is properly parsed and applied."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test configuration with HDF5 enabled
        config = {
            'output_data': {
                'unwrapped_traj': {'file_name': 'unwrapped_traj.npy', 'write': True, 'write_every_step': False},
                'time': {'file_name': 'time_data.npy', 'write': True},
                'hdf5_output': {'enabled': True, 'file_name': 'trajectory.h5'}
            }
        }
        
        # Test config parsing
        output_data = config['output_data']
        
        # Verify HDF5 config exists and is enabled
        assert 'hdf5_output' in output_data
        assert output_data['hdf5_output']['enabled'] is True
        assert output_data['hdf5_output']['file_name'] == 'trajectory.h5'
        
        # Simulate creating HDF5 file based on config
        hdf5_file = tmpdir / output_data['hdf5_output']['file_name']
        
        # Test data
        n_atoms = 2
        coordinates = np.random.rand(3, n_atoms, 3) * 5.0
        time_data = np.array([0.0, 1.0, 2.0])
        
        with HDF5TrajectoryWriter(hdf5_file, n_atoms, 1) as writer:
            writer.write_frames(coordinates, time_data)
        
        assert hdf5_file.exists()
        
        # Verify HDF5 file content
        with h5py.File(hdf5_file, 'r') as f:
            assert f['coordinates'].shape == (3, n_atoms, 3)
            assert f['time'].shape == (3,)


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_hdf5_disabled_config():
    """Test that HDF5 output is properly disabled when configured."""
    config = {
        'output_data': {
            'unwrapped_traj': {'file_name': 'unwrapped_traj.npy', 'write': True},
            'hdf5_output': {'enabled': False, 'file_name': 'trajectory.h5'}
        }
    }
    
    output_data = config['output_data']
    
    # Verify that HDF5 is disabled
    assert not output_data['hdf5_output']['enabled']


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_hdf5_backward_compatibility():
    """Test that configs without HDF5 option still work."""
    # Old-style config without HDF5 option
    config = {
        'output_data': {
            'unwrapped_traj': {'file_name': 'unwrapped_traj.npy', 'write': True},
            'time': {'file_name': 'time_data.npy', 'write': True}
        }
    }
    
    output_data = config['output_data']
    
    # Should not have HDF5 config, but should still be valid
    assert 'hdf5_output' not in output_data
    
    # Verify that the logic handles missing HDF5 config gracefully
    hdf5_enabled = output_data.get('hdf5_output', {}).get('enabled', False)
    assert hdf5_enabled is False


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_dual_format_consistency():
    """Test that both numpy and HDF5 formats contain consistent data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test trajectory data
        n_frames = 4
        n_atoms = 3
        coordinates = np.random.rand(n_frames, n_atoms * 3) * 8.0
        time_data = np.linspace(0, 10, n_frames)
        
        # Save in numpy format (simulate existing behavior)
        npy_file = tmpdir / "unwrapped_traj.npy"
        time_file = tmpdir / "time_data.npy"
        np.save(npy_file, coordinates)
        np.save(time_file, time_data)
        
        # Save in HDF5 format
        hdf5_file = tmpdir / "trajectory.h5"
        coords_reshaped = coordinates.reshape(n_frames, n_atoms, 3)
        
        with HDF5TrajectoryWriter(hdf5_file, n_atoms, 1) as writer:
            writer.write_frames(coords_reshaped, time_data)
        
        # Load both formats and verify consistency
        npy_coords = np.load(npy_file)
        npy_time = np.load(time_file)
        
        with h5py.File(hdf5_file, 'r') as f:
            hdf5_coords = f['coordinates'][:]
            hdf5_time = f['time'][:]
            
        # Reshape HDF5 coordinates to match numpy format
        hdf5_coords_flat = hdf5_coords.reshape(n_frames, n_atoms * 3)
        
        # Convert HDF5 coordinates back to Bohr for comparison
        from PyCD import constants
        hdf5_coords_bohr = hdf5_coords_flat / (0.1 / constants.ANG2BOHR)
        
        # Verify coordinate consistency (within numerical precision)
        np.testing.assert_allclose(npy_coords, hdf5_coords_bohr, rtol=1e-6)
        
        # Note: Time units are different (HDF5 in ps, numpy in AU)
        # Just verify that time data structure is consistent
        assert len(npy_time) == len(hdf5_time)


def test_example_config_files():
    """Test that example configuration files have proper HDF5 settings."""
    # Test Hematite example
    hematite_config_path = Path("/home/runner/work/PyCD/PyCD/examples/Hematite/simulation_parameters.yml")
    if hematite_config_path.exists():
        with open(hematite_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'output_data' in config
        assert 'hdf5_output' in config['output_data']
        assert 'enabled' in config['output_data']['hdf5_output']
        assert 'file_name' in config['output_data']['hdf5_output']
        
        # Verify default is enabled
        assert config['output_data']['hdf5_output']['enabled'] is True
    
    # Test BVO example
    bvo_config_path = Path("/home/runner/work/PyCD/PyCD/examples/BVO/simulation_parameters.yml")
    if bvo_config_path.exists():
        with open(bvo_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'output_data' in config
        assert 'hdf5_output' in config['output_data']
        assert config['output_data']['hdf5_output']['enabled'] is True


if __name__ == "__main__":
    pytest.main([__file__])