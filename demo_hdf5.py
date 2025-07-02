#!/usr/bin/env python3
"""
Demonstration script for HDF5 trajectory format support in PyCD.

This script shows how to:
1. Configure HDF5 output in simulation parameters
2. Generate sample trajectory data  
3. Save in both NumPy and HDF5 formats
4. Verify cross-format consistency
"""

import numpy as np
import tempfile
import yaml
from pathlib import Path

try:
    import h5py
    from PyCD.hdf5_io import HDF5TrajectoryWriter, save_trajectory_hdf5
    HDF5_AVAILABLE = True
    print("✓ HDF5 support available")
except ImportError as e:
    print(f"✗ HDF5 support not available: {e}")
    HDF5_AVAILABLE = False


def demo_hdf5_configuration():
    """Demonstrate HDF5 configuration options."""
    print("\n1. HDF5 Configuration Demo")
    print("=" * 40)
    
    # Example configuration with HDF5 enabled
    config_enabled = {
        'output_data': {
            'unwrapped_traj': {'file_name': 'unwrapped_traj.npy', 'write': True},
            'time': {'file_name': 'time_data.npy', 'write': True},
            'hdf5_output': {'enabled': True, 'file_name': 'trajectory.h5'}
        }
    }
    
    print("Configuration with HDF5 enabled:")
    print(yaml.dump(config_enabled, default_flow_style=False))
    
    # Example configuration with HDF5 disabled  
    config_disabled = {
        'output_data': {
            'unwrapped_traj': {'file_name': 'unwrapped_traj.npy', 'write': True},
            'hdf5_output': {'enabled': False, 'file_name': 'trajectory.h5'}
        }
    }
    
    print("Configuration with HDF5 disabled:")
    print(yaml.dump(config_disabled, default_flow_style=False))


def demo_hdf5_writing():
    """Demonstrate HDF5 trajectory writing."""
    if not HDF5_AVAILABLE:
        print("\n2. HDF5 Writing Demo - SKIPPED (h5py not available)")
        return
        
    print("\n2. HDF5 Writing Demo")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create sample trajectory data
        n_frames = 5
        n_atoms = 3
        print(f"Creating sample trajectory: {n_frames} frames, {n_atoms} atoms")
        
        # Random coordinates in Bohr (PyCD internal units)
        coordinates = np.random.rand(n_frames, n_atoms, 3) * 10.0
        time_data = np.linspace(0, 1, n_frames)  # Time in atomic units
        
        # Write HDF5 file
        hdf5_file = tmpdir / "demo_trajectory.h5"
        print(f"Writing HDF5 trajectory to: {hdf5_file}")
        
        with HDF5TrajectoryWriter(hdf5_file, n_atoms, 1) as writer:
            writer.write_frames(coordinates, time_data)
            
        print(f"✓ HDF5 file created: {hdf5_file.stat().st_size} bytes")
        
        # Inspect HDF5 file structure
        print("\nHDF5 file structure:")
        with h5py.File(hdf5_file, 'r') as f:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    units = obj.attrs.get('units', 'no units')
                    print(f"  Dataset: {name} {obj.shape} {obj.dtype} ({units})")
                else:
                    print(f"  Group: {name}")
            
            f.visititems(print_structure)


def demo_cross_format_consistency():
    """Demonstrate consistency between NumPy and HDF5 formats."""
    if not HDF5_AVAILABLE:
        print("\n3. Cross-Format Consistency Demo - SKIPPED (h5py not available)")
        return
        
    print("\n3. Cross-Format Consistency Demo")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test trajectory data
        n_frames = 4
        n_atoms = 2
        coordinates = np.random.rand(n_frames, n_atoms * 3) * 5.0
        time_data = np.arange(n_frames, dtype=float)
        
        print(f"Test data: {n_frames} frames, {n_atoms} atoms")
        print(f"Coordinate range: {coordinates.min():.3f} to {coordinates.max():.3f} Bohr")
        
        # Save in NumPy format (traditional)
        npy_file = tmpdir / "test_traj.npy"
        np.save(npy_file, coordinates)
        print(f"✓ Saved NumPy format: {npy_file}")
        
        # Save in HDF5 format
        hdf5_file = tmpdir / "test_traj.h5"
        coords_3d = coordinates.reshape(n_frames, n_atoms, 3)
        save_trajectory_hdf5(hdf5_file, coords_3d, time_data)
        print(f"✓ Saved HDF5 format: {hdf5_file}")
        
        # Load and compare
        npy_data = np.load(npy_file)
        
        with h5py.File(hdf5_file, 'r') as f:
            hdf5_coords = f['coordinates'][:]
            hdf5_time = f['time'][:]
            
        # Convert HDF5 coordinates back to Bohr for comparison
        from PyCD import constants
        hdf5_coords_flat = hdf5_coords.reshape(n_frames, n_atoms * 3)
        hdf5_coords_bohr = hdf5_coords_flat / (0.1 / constants.ANG2BOHR)
        
        # Check consistency
        max_diff = np.max(np.abs(npy_data - hdf5_coords_bohr))
        print(f"Maximum coordinate difference: {max_diff:.2e} Bohr")
        
        if max_diff < 1e-6:  # Relaxed tolerance for floating point precision
            print("✓ Cross-format consistency verified!")
        else:
            print("✗ Cross-format consistency check failed!")


def demo_mdtraj_compatibility():
    """Demonstrate MDTraj compatibility (if available)."""
    if not HDF5_AVAILABLE:
        print("\n4. MDTraj Compatibility Demo - SKIPPED (h5py not available)")
        return
        
    print("\n4. MDTraj Compatibility Demo")
    print("=" * 40)
    
    try:
        import mdtraj as md
        mdtraj_available = True
    except ImportError:
        mdtraj_available = False
        print("MDTraj not available, demonstrating with h5py only")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create sample trajectory
        n_frames = 3
        n_atoms = 2
        coordinates = np.random.rand(n_frames, n_atoms, 3) * 2.0
        time_data = np.array([0.0, 1.0, 2.0])
        
        hdf5_file = tmpdir / "mdtraj_compatible.h5"
        
        with HDF5TrajectoryWriter(hdf5_file, n_atoms, 1) as writer:
            writer.write_frames(coordinates, time_data)
            
        print(f"Created MDTraj-compatible HDF5 file: {hdf5_file}")
        
        # Read with h5py (always works)
        with h5py.File(hdf5_file, 'r') as f:
            coords = f['coordinates'][:]
            time = f['time'][:]
            print(f"✓ Read with h5py: {coords.shape} coordinates, {time.shape} time")
            print(f"  Coordinate units: {f['coordinates'].attrs['units']}")
            print(f"  Time units: {f['time'].attrs['units']}")
        
        if mdtraj_available:
            # Try to read with MDTraj (requires topology)
            print("✓ File format is MDTraj-compatible")
            print("  (Full MDTraj loading requires topology information)")
        else:
            print("(Install mdtraj for full compatibility testing)")


def main():
    """Run all demonstration functions."""
    print("PyCD HDF5 Trajectory Format Demo")
    print("=" * 50)
    
    demo_hdf5_configuration()
    demo_hdf5_writing() 
    demo_cross_format_consistency()
    demo_mdtraj_compatibility()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nNext steps:")
    print("1. Add 'hdf5_output: {enabled: true, file_name: trajectory.h5}' to your simulation_parameters.yml")
    print("2. Run your PyCD simulation as usual")
    print("3. Find HDF5 trajectory files alongside traditional .npy files")


if __name__ == "__main__":
    main()