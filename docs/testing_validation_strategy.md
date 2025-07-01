# Testing and Validation Strategy for HDF5 Trajectory Migration

## Overview

This document outlines a comprehensive testing and validation strategy to ensure data integrity, performance, and reliability during the transition from NumPy `.npy` to HDF5 trajectory storage format in PyCD.

## 1. Testing Framework Architecture

### 1.1 Test Categories

```python
# Test organization structure
PyCD/tests/
├── test_trajectory_formats.py          # Core format testing
├── test_dual_writing.py               # Dual-writing functionality
├── test_hdf5_schema.py                # HDF5 schema validation
├── test_format_conversion.py          # Format conversion utilities
├── test_performance_comparison.py     # Performance benchmarks
├── test_backward_compatibility.py     # Compatibility testing
└── fixtures/
    ├── sample_trajectories/           # Test trajectory data
    ├── configurations/                # Test configurations
    └── reference_data/                # Golden reference data
```

### 1.2 Test Data Management

```python
# Fixtures for consistent test data
@pytest.fixture
def sample_trajectory_data():
    """Generate consistent test trajectory data"""
    n_frames = 1000
    n_species = 2
    
    # Create deterministic trajectory with known properties
    np.random.seed(42)
    
    # Unwrapped coordinates with drift
    unwrapped = np.zeros((n_frames, n_species * 3))
    for frame in range(1, n_frames):
        # Add small random displacement + drift
        drift = np.array([0.01, 0.005, 0.0] * n_species) * frame
        noise = np.random.normal(0, 0.1, n_species * 3)
        unwrapped[frame] = unwrapped[frame-1] + drift + noise
    
    # Wrapped coordinates (apply PBC)
    box_size = np.array([10.0, 10.0, 12.0])
    wrapped = unwrapped % np.tile(box_size, n_species)
    
    # Time data
    dt = 1e-12
    time = np.arange(n_frames) * dt
    
    # Energy data with realistic fluctuations
    base_energy = -125.3
    energy = base_energy + np.random.normal(0, 0.05, n_frames)
    
    return {
        'unwrapped': unwrapped,
        'wrapped': wrapped,
        'time': time,
        'energy': energy,
        'metadata': {
            'n_frames': n_frames,
            'n_species': n_species,
            'box_size': box_size,
            'dt': dt
        }
    }

@pytest.fixture
def simulation_metadata():
    """Standard simulation metadata for testing"""
    return {
        'version': '1.0.0',
        'created_on': '2024-01-01T00:00:00Z',
        'pycd_version': '1.0.0',
        'lattice_matrix': np.eye(3) * 10.0,
        'pbc': np.array([1, 1, 1]),
        'system_size': np.array([2, 2, 1]),
        'species_count': np.array([5, 5]),
        'total_species': 10,
        'temperature': 300.0,
        'time_interval': 1e-12,
        't_final': 1e-9,
        'n_traj': 5,
        'random_seed': 12345
    }
```

## 2. Data Integrity Testing

### 2.1 Cross-Format Validation Tests

```python
class TestDataIntegrity:
    """Test data integrity between .npy and HDF5 formats"""
    
    def test_coordinate_data_equivalence(self, sample_trajectory_data, tmp_path):
        """Test that coordinate data is identical between formats"""
        
        # Write data in both formats
        npy_path = tmp_path / "npy_format"
        hdf5_path = tmp_path / "trajectory.h5"
        
        # Write .npy files
        npy_path.mkdir()
        np.save(npy_path / "unwrapped_traj.npy", sample_trajectory_data['unwrapped'])
        np.save(npy_path / "wrapped_traj.npy", sample_trajectory_data['wrapped'])
        np.save(npy_path / "time.npy", sample_trajectory_data['time'])
        np.save(npy_path / "energy.npy", sample_trajectory_data['energy'])
        
        # Write HDF5 file
        metadata = sample_trajectory_data['metadata']
        with h5py.File(hdf5_path, 'w') as f:
            # Create trajectory group
            traj = f.create_group('/trajectories/traj_001')
            coords = traj.create_group('coordinates')
            
            coords.create_dataset('unwrapped', data=sample_trajectory_data['unwrapped'])
            coords.create_dataset('wrapped', data=sample_trajectory_data['wrapped'])
            traj.create_dataset('time', data=sample_trajectory_data['time'])
            traj.create_dataset('energy', data=sample_trajectory_data['energy'])
        
        # Load and compare
        npy_unwrapped = np.load(npy_path / "unwrapped_traj.npy")
        
        with h5py.File(hdf5_path, 'r') as f:
            h5_unwrapped = f['/trajectories/traj_001/coordinates/unwrapped'][:]
        
        # Verify exact equivalence
        assert np.array_equal(npy_unwrapped, h5_unwrapped), \
            "Unwrapped coordinates differ between formats"
        
        # Test other data types
        npy_time = np.load(npy_path / "time.npy")
        with h5py.File(hdf5_path, 'r') as f:
            h5_time = f['/trajectories/traj_001/time'][:]
        
        assert np.array_equal(npy_time, h5_time), \
            "Time data differs between formats"
    
    def test_floating_point_precision(self, tmp_path):
        """Test that floating point precision is preserved"""
        
        # Create data with specific precision requirements
        test_data = np.array([
            1.23456789012345e-15,  # Very small number
            1.23456789012345e+15,  # Very large number
            np.pi,                 # Irrational number
            np.e,                  # Another irrational
            1.0/3.0,              # Repeating decimal
        ])
        
        # Write to both formats
        npy_file = tmp_path / "precision_test.npy"
        h5_file = tmp_path / "precision_test.h5"
        
        np.save(npy_file, test_data)
        
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('test_data', data=test_data, dtype=np.float64)
        
        # Load and compare
        npy_loaded = np.load(npy_file)
        
        with h5py.File(h5_file, 'r') as f:
            h5_loaded = f['test_data'][:]
        
        # Should be exactly equal for same precision
        assert np.array_equal(npy_loaded, h5_loaded), \
            "Floating point precision not preserved"
    
    def test_large_trajectory_integrity(self, tmp_path):
        """Test integrity for large trajectory files"""
        
        # Create large trajectory (10M frames)
        n_frames = 10_000_000
        n_species = 1
        
        # Generate data in chunks to avoid memory issues
        chunk_size = 100_000
        
        npy_file = tmp_path / "large_traj.npy"
        h5_file = tmp_path / "large_traj.h5"
        
        # Write NPY in chunks
        for chunk_start in range(0, n_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_frames)
            chunk_data = np.random.random((chunk_end - chunk_start, n_species * 3))
            
            if chunk_start == 0:
                np.save(npy_file, chunk_data)
            else:
                # Append to existing file
                existing_data = np.load(npy_file)
                combined_data = np.vstack([existing_data, chunk_data])
                np.save(npy_file, combined_data)
        
        # Write HDF5 with chunking
        with h5py.File(h5_file, 'w') as f:
            dataset = f.create_dataset(
                'trajectory',
                shape=(0, n_species * 3),
                maxshape=(None, n_species * 3),
                chunks=(chunk_size, n_species * 3),
                dtype=np.float64
            )
            
            # Write in chunks
            for chunk_start in range(0, n_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_frames)
                chunk_data = np.random.random((chunk_end - chunk_start, n_species * 3))
                
                old_size = dataset.shape[0]
                dataset.resize((old_size + chunk_data.shape[0], n_species * 3))
                dataset[old_size:] = chunk_data
        
        # Statistical comparison (exact comparison too memory intensive)
        npy_data = np.load(npy_file)
        
        with h5py.File(h5_file, 'r') as f:
            h5_data = f['trajectory']
            
            # Compare shapes
            assert npy_data.shape == h5_data.shape, \
                f"Shape mismatch: NPY {npy_data.shape} vs HDF5 {h5_data.shape}"
            
            # Compare statistical properties in chunks
            for chunk_start in range(0, n_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_frames)
                
                npy_chunk = npy_data[chunk_start:chunk_end]
                h5_chunk = h5_data[chunk_start:chunk_end]
                
                # Compare means and standard deviations
                np.testing.assert_allclose(
                    np.mean(npy_chunk, axis=0),
                    np.mean(h5_chunk, axis=0),
                    rtol=1e-10
                )
                np.testing.assert_allclose(
                    np.std(npy_chunk, axis=0),
                    np.std(h5_chunk, axis=0),
                    rtol=1e-10
                )
```

### 2.2 Schema Validation Tests

```python
class TestHDF5Schema:
    """Test HDF5 schema compliance and validation"""
    
    def test_required_metadata_present(self, simulation_metadata, tmp_path):
        """Test that all required metadata is present"""
        
        h5_file = tmp_path / "schema_test.h5"
        
        # Create file with metadata
        with h5py.File(h5_file, 'w') as f:
            # Write metadata
            meta = f.create_group('/metadata')
            sim_info = meta.create_group('simulation_info')
            sim_info.attrs['version'] = simulation_metadata['version']
            sim_info.attrs['created_on'] = simulation_metadata['created_on']
            sim_info.attrs['format_version'] = 'hdf5_v1'
            
        # Validate metadata
        with h5py.File(h5_file, 'r') as f:
            meta = f['/metadata']
            
            # Check required groups
            assert 'simulation_info' in meta, "Missing simulation_info group"
            
            # Check required attributes
            sim_info = meta['simulation_info']
            required_attrs = ['version', 'created_on', 'format_version']
            for attr in required_attrs:
                assert attr in sim_info.attrs, f"Missing required attribute: {attr}"
    
    def test_dataset_attributes(self, sample_trajectory_data, tmp_path):
        """Test that datasets have proper attributes"""
        
        h5_file = tmp_path / "attributes_test.h5"
        
        with h5py.File(h5_file, 'w') as f:
            traj = f.create_group('/trajectories/traj_001')
            coords = traj.create_group('coordinates')
            
            # Create unwrapped coordinates with attributes
            unwrapped = coords.create_dataset(
                'unwrapped',
                data=sample_trajectory_data['unwrapped']
            )
            unwrapped.attrs['units'] = 'bohr'
            unwrapped.attrs['description'] = 'Unwrapped coordinates'
            unwrapped.attrs['species_layout'] = 'interleaved_xyz'
            
        # Validate attributes
        with h5py.File(h5_file, 'r') as f:
            unwrapped = f['/trajectories/traj_001/coordinates/unwrapped']
            
            required_attrs = ['units', 'description', 'species_layout']
            for attr in required_attrs:
                assert attr in unwrapped.attrs, \
                    f"Missing required attribute: {attr}"
            
            # Check attribute values
            assert unwrapped.attrs['units'].decode() == 'bohr'
            assert unwrapped.attrs['species_layout'].decode() == 'interleaved_xyz'
    
    def test_data_type_consistency(self, sample_trajectory_data, tmp_path):
        """Test that data types are consistent with schema"""
        
        h5_file = tmp_path / "dtype_test.h5"
        
        with h5py.File(h5_file, 'w') as f:
            traj = f.create_group('/trajectories/traj_001')
            
            # Coordinates should be float64
            coords = traj.create_group('coordinates')
            unwrapped = coords.create_dataset(
                'unwrapped',
                data=sample_trajectory_data['unwrapped'],
                dtype=np.float64
            )
            
            # Time should be float64
            time_ds = traj.create_dataset(
                'time',
                data=sample_trajectory_data['time'],
                dtype=np.float64
            )
            
            # Energy should be float64
            energy_ds = traj.create_dataset(
                'energy',
                data=sample_trajectory_data['energy'],
                dtype=np.float64
            )
        
        # Validate data types
        with h5py.File(h5_file, 'r') as f:
            assert f['/trajectories/traj_001/coordinates/unwrapped'].dtype == np.float64
            assert f['/trajectories/traj_001/time'].dtype == np.float64
            assert f['/trajectories/traj_001/energy'].dtype == np.float64
```

## 3. Performance Testing

### 3.1 Benchmark Suite

```python
class TestPerformance:
    """Performance benchmarks for trajectory formats"""
    
    def test_write_performance_comparison(self, sample_trajectory_data, tmp_path):
        """Compare write performance between formats"""
        
        import time
        
        # Test data
        n_frames = 10000
        n_species = 5
        trajectory_data = np.random.random((n_frames, n_species * 3))
        
        # Time NPY writing
        npy_file = tmp_path / "perf_test.npy"
        start_time = time.time()
        np.save(npy_file, trajectory_data)
        npy_write_time = time.time() - start_time
        
        # Time HDF5 writing (uncompressed)
        h5_file_uncompressed = tmp_path / "perf_test_uncompressed.h5"
        start_time = time.time()
        with h5py.File(h5_file_uncompressed, 'w') as f:
            f.create_dataset('trajectory', data=trajectory_data)
        h5_uncompressed_time = time.time() - start_time
        
        # Time HDF5 writing (compressed)
        h5_file_compressed = tmp_path / "perf_test_compressed.h5"
        start_time = time.time()
        with h5py.File(h5_file_compressed, 'w') as f:
            f.create_dataset(
                'trajectory',
                data=trajectory_data,
                compression='gzip',
                compression_opts=6
            )
        h5_compressed_time = time.time() - start_time
        
        # Log results for analysis
        print(f"NPY write time: {npy_write_time:.3f}s")
        print(f"HDF5 uncompressed write time: {h5_uncompressed_time:.3f}s")
        print(f"HDF5 compressed write time: {h5_compressed_time:.3f}s")
        
        # Performance assertions (reasonable bounds)
        assert h5_uncompressed_time < npy_write_time * 5, \
            "HDF5 uncompressed writing too slow"
        assert h5_compressed_time < npy_write_time * 10, \
            "HDF5 compressed writing too slow"
    
    def test_read_performance_comparison(self, tmp_path):
        """Compare read performance between formats"""
        
        import time
        
        # Create test data
        n_frames = 50000
        n_species = 3
        trajectory_data = np.random.random((n_frames, n_species * 3))
        
        # Save in both formats
        npy_file = tmp_path / "read_perf_test.npy"
        h5_file = tmp_path / "read_perf_test.h5"
        
        np.save(npy_file, trajectory_data)
        
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset(
                'trajectory',
                data=trajectory_data,
                compression='gzip',
                compression_opts=6,
                chunks=True
            )
        
        # Time NPY reading
        start_time = time.time()
        npy_data = np.load(npy_file)
        npy_read_time = time.time() - start_time
        
        # Time HDF5 reading (full dataset)
        start_time = time.time()
        with h5py.File(h5_file, 'r') as f:
            h5_data = f['trajectory'][:]
        h5_read_time = time.time() - start_time
        
        # Time HDF5 partial reading (advantage of HDF5)
        start_time = time.time()
        with h5py.File(h5_file, 'r') as f:
            h5_partial = f['trajectory'][1000:2000]
        h5_partial_read_time = time.time() - start_time
        
        print(f"NPY read time: {npy_read_time:.3f}s")
        print(f"HDF5 read time: {h5_read_time:.3f}s")
        print(f"HDF5 partial read time: {h5_partial_read_time:.3f}s")
        
        # Verify data integrity
        np.testing.assert_array_equal(npy_data, h5_data)
        
        # Partial reading should be much faster
        assert h5_partial_read_time < h5_read_time / 5, \
            "HDF5 partial reading not significantly faster"
    
    def test_storage_efficiency(self, tmp_path):
        """Compare storage efficiency between formats"""
        
        import os
        
        # Create test data with different characteristics
        n_frames = 20000
        n_species = 4
        
        # Regular trajectory data
        regular_data = np.random.random((n_frames, n_species * 3))
        
        # Sparse data (mostly zeros)
        sparse_data = np.zeros((n_frames, n_species * 3))
        sparse_data[::100] = np.random.random((n_frames//100, n_species * 3))
        
        # Test regular data compression
        for name, data in [("regular", regular_data), ("sparse", sparse_data)]:
            
            # NPY file
            npy_file = tmp_path / f"{name}_data.npy"
            np.save(npy_file, data)
            npy_size = os.path.getsize(npy_file)
            
            # HDF5 uncompressed
            h5_uncompressed = tmp_path / f"{name}_uncompressed.h5"
            with h5py.File(h5_uncompressed, 'w') as f:
                f.create_dataset('data', data=data)
            h5_uncompressed_size = os.path.getsize(h5_uncompressed)
            
            # HDF5 compressed
            h5_compressed = tmp_path / f"{name}_compressed.h5"
            with h5py.File(h5_compressed, 'w') as f:
                f.create_dataset(
                    'data',
                    data=data,
                    compression='gzip',
                    compression_opts=6,
                    chunks=True,
                    shuffle=True
                )
            h5_compressed_size = os.path.getsize(h5_compressed)
            
            # Calculate compression ratios
            compression_ratio = npy_size / h5_compressed_size
            
            print(f"{name.title()} data storage sizes:")
            print(f"  NPY: {npy_size/1024/1024:.1f} MB")
            print(f"  HDF5 uncompressed: {h5_uncompressed_size/1024/1024:.1f} MB")
            print(f"  HDF5 compressed: {h5_compressed_size/1024/1024:.1f} MB")
            print(f"  Compression ratio: {compression_ratio:.1f}x")
            
            # Sparse data should compress very well
            if name == "sparse":
                assert compression_ratio > 5, \
                    f"Poor compression for sparse data: {compression_ratio:.1f}x"
```

## 4. Incremental Writing Tests

### 4.1 Append Operations

```python
class TestIncrementalWriting:
    """Test incremental writing capabilities"""
    
    def test_hdf5_append_operations(self, tmp_path):
        """Test that HDF5 datasets can be extended properly"""
        
        h5_file = tmp_path / "append_test.h5"
        n_species = 2
        
        with h5py.File(h5_file, 'w') as f:
            # Create resizable dataset
            dataset = f.create_dataset(
                'trajectory',
                shape=(0, n_species * 3),
                maxshape=(None, n_species * 3),
                chunks=(1000, n_species * 3),
                dtype=np.float64
            )
            
            # Append data in chunks
            chunk_size = 500
            total_frames = 2500
            
            for chunk_start in range(0, total_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_frames)
                chunk_frames = chunk_end - chunk_start
                
                # Generate chunk data
                chunk_data = np.random.random((chunk_frames, n_species * 3))
                
                # Resize and append
                old_size = dataset.shape[0]
                dataset.resize((old_size + chunk_frames, n_species * 3))
                dataset[old_size:] = chunk_data
        
        # Verify final size
        with h5py.File(h5_file, 'r') as f:
            dataset = f['trajectory']
            assert dataset.shape[0] == total_frames, \
                f"Incorrect final size: {dataset.shape[0]} vs {total_frames}"
    
    def test_concurrent_trajectory_writing(self, tmp_path):
        """Test writing multiple trajectories simultaneously"""
        
        h5_file = tmp_path / "concurrent_test.h5"
        n_trajectories = 5
        n_species = 2
        frames_per_traj = 1000
        
        with h5py.File(h5_file, 'w') as f:
            # Create trajectory groups and datasets
            trajectories = {}
            for traj_id in range(n_trajectories):
                traj_name = f"traj_{traj_id:03d}"
                traj_group = f.create_group(f'/trajectories/{traj_name}')
                
                coords_group = traj_group.create_group('coordinates')
                unwrapped = coords_group.create_dataset(
                    'unwrapped',
                    shape=(0, n_species * 3),
                    maxshape=(None, n_species * 3),
                    chunks=(100, n_species * 3),
                    dtype=np.float64
                )
                
                time_ds = traj_group.create_dataset(
                    'time',
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(100,),
                    dtype=np.float64
                )
                
                trajectories[traj_id] = {
                    'coords': unwrapped,
                    'time': time_ds,
                    'frame_count': 0
                }
            
            # Simulate concurrent writing (round-robin)
            for frame in range(frames_per_traj):
                for traj_id in range(n_trajectories):
                    traj = trajectories[traj_id]
                    
                    # Generate frame data
                    coord_data = np.random.random((1, n_species * 3))
                    time_data = np.array([frame * 1e-12])
                    
                    # Append to datasets
                    old_coord_size = traj['coords'].shape[0]
                    traj['coords'].resize((old_coord_size + 1, n_species * 3))
                    traj['coords'][old_coord_size:] = coord_data
                    
                    old_time_size = traj['time'].shape[0]
                    traj['time'].resize((old_time_size + 1,))
                    traj['time'][old_time_size:] = time_data
                    
                    traj['frame_count'] += 1
        
        # Verify all trajectories have correct size
        with h5py.File(h5_file, 'r') as f:
            for traj_id in range(n_trajectories):
                traj_name = f"traj_{traj_id:03d}"
                coords = f[f'/trajectories/{traj_name}/coordinates/unwrapped']
                time_ds = f[f'/trajectories/{traj_name}/time']
                
                assert coords.shape[0] == frames_per_traj, \
                    f"Incorrect coord size for {traj_name}"
                assert time_ds.shape[0] == frames_per_traj, \
                    f"Incorrect time size for {traj_name}"
```

## 5. Error Handling and Recovery Tests

### 5.1 Robustness Testing

```python
class TestErrorHandling:
    """Test error handling and recovery mechanisms"""
    
    def test_corrupted_file_handling(self, tmp_path):
        """Test handling of corrupted HDF5 files"""
        
        h5_file = tmp_path / "corrupted_test.h5"
        
        # Create a valid file first
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('test_data', data=np.random.random((100, 3)))
        
        # Corrupt the file by truncating it
        with open(h5_file, 'r+b') as f:
            f.seek(100)  # Seek to position 100
            f.truncate()  # Truncate from this position
        
        # Test that we can detect corruption
        with pytest.raises((OSError, h5py.h5f.FileCloseDegreeError)):
            with h5py.File(h5_file, 'r') as f:
                data = f['test_data'][:]
    
    def test_disk_space_handling(self, tmp_path):
        """Test handling of insufficient disk space"""
        
        # This test is challenging to implement in a platform-independent way
        # In practice, would need to mock filesystem operations
        # For now, document the expected behavior
        
        h5_file = tmp_path / "disk_space_test.h5"
        
        # Expected behavior when disk space runs out:
        # 1. HDF5 writing should raise OSError
        # 2. Partial data should be recoverable
        # 3. File should not be left in corrupted state
        
        # Placeholder test - would need platform-specific implementation
        pytest.skip("Disk space testing requires platform-specific implementation")
    
    def test_memory_limitation_handling(self, tmp_path):
        """Test handling of memory limitations during large operations"""
        
        h5_file = tmp_path / "memory_test.h5"
        
        # Test writing very large datasets in chunks
        # This tests the chunking strategy
        
        with h5py.File(h5_file, 'w') as f:
            # Create a dataset that would be large if fully loaded
            n_frames = 1_000_000  # 1M frames
            n_coords = 30  # 10 species * 3 coords
            
            dataset = f.create_dataset(
                'large_trajectory',
                shape=(n_frames, n_coords),
                chunks=(10000, n_coords),  # 10k frame chunks
                dtype=np.float64
            )
            
            # Write in small chunks to avoid memory issues
            chunk_size = 1000
            for start in range(0, n_frames, chunk_size):
                end = min(start + chunk_size, n_frames)
                chunk_data = np.random.random((end - start, n_coords))
                dataset[start:end] = chunk_data
                
                # Simulate memory constraint check
                # In practice, would monitor memory usage
                if start % 100000 == 0:  # Every 100k frames
                    print(f"Written {start} frames")
        
        # Verify file integrity
        with h5py.File(h5_file, 'r') as f:
            dataset = f['large_trajectory']
            assert dataset.shape[0] == n_frames, \
                "Dataset size incorrect after chunked writing"
            
            # Test random access (benefit of chunking)
            sample_data = dataset[500000:500010]  # Sample middle section
            assert sample_data.shape == (10, n_coords), \
                "Random access failed"
```

## 6. Backward Compatibility Tests

### 6.1 Legacy Format Support

```python
class TestBackwardCompatibility:
    """Test backward compatibility with existing .npy workflows"""
    
    def test_existing_analysis_scripts(self, sample_trajectory_data, tmp_path):
        """Test that existing analysis scripts continue to work"""
        
        # Create .npy files in expected format
        traj_dir = tmp_path / "traj1"
        traj_dir.mkdir()
        
        np.save(traj_dir / "unwrapped_traj.npy", sample_trajectory_data['unwrapped'])
        np.save(traj_dir / "wrapped_traj.npy", sample_trajectory_data['wrapped'])
        np.save(traj_dir / "time.npy", sample_trajectory_data['time'])
        np.save(traj_dir / "energy.npy", sample_trajectory_data['energy'])
        
        # Test that existing unwrap_trajectory_file function works
        from PyCD.core import unwrap_trajectory_file
        
        system_size = np.array([2, 2, 1])
        lattice_matrix = np.eye(3) * 10.0
        pbc = [1, 1, 1]
        
        # This should work without modification
        result = unwrap_trajectory_file(
            traj_dir / "wrapped_traj.npy",
            system_size,
            lattice_matrix,
            pbc
        )
        
        assert result is not None, "Existing unwrap function failed"
        assert result.shape == sample_trajectory_data['wrapped'].shape, \
            "Unwrap result shape incorrect"
    
    def test_format_conversion_utilities(self, sample_trajectory_data, simulation_metadata, tmp_path):
        """Test conversion between .npy and HDF5 formats"""
        
        # Create .npy trajectory files
        npy_dir = tmp_path / "npy_trajectory"
        npy_dir.mkdir()
        
        np.save(npy_dir / "unwrapped_traj.npy", sample_trajectory_data['unwrapped'])
        np.save(npy_dir / "wrapped_traj.npy", sample_trajectory_data['wrapped'])
        np.save(npy_dir / "time.npy", sample_trajectory_data['time'])
        np.save(npy_dir / "energy.npy", sample_trajectory_data['energy'])
        
        # Convert to HDF5
        h5_file = tmp_path / "converted_trajectory.h5"
        
        # This function would be implemented as part of the migration
        def convert_npy_to_hdf5(npy_directory, hdf5_file, metadata):
            """Convert .npy trajectory to HDF5 format"""
            
            with h5py.File(hdf5_file, 'w') as f:
                # Write metadata
                meta = f.create_group('/metadata')
                sim_info = meta.create_group('simulation_info')
                for key, value in metadata.items():
                    if isinstance(value, np.ndarray):
                        if key in ['lattice_matrix', 'pbc', 'system_size', 'species_count']:
                            sys_info = meta.create_group('system_info') if 'system_info' not in meta else meta['system_info']
                            sys_info.create_dataset(key, data=value)
                    else:
                        sim_info.attrs[key] = value
                
                # Write trajectory data
                traj = f.create_group('/trajectories/traj_001')
                coords = traj.create_group('coordinates')
                
                # Load and write coordinate data
                unwrapped_data = np.load(npy_directory / "unwrapped_traj.npy")
                wrapped_data = np.load(npy_directory / "wrapped_traj.npy")
                time_data = np.load(npy_directory / "time.npy")
                energy_data = np.load(npy_directory / "energy.npy")
                
                coords.create_dataset('unwrapped', data=unwrapped_data,
                                     compression='gzip', compression_opts=6)
                coords.create_dataset('wrapped', data=wrapped_data,
                                     compression='gzip', compression_opts=6)
                traj.create_dataset('time', data=time_data,
                                   compression='gzip', compression_opts=6)
                traj.create_dataset('energy', data=energy_data,
                                   compression='gzip', compression_opts=6)
        
        # Perform conversion
        convert_npy_to_hdf5(npy_dir, h5_file, simulation_metadata)
        
        # Verify conversion
        with h5py.File(h5_file, 'r') as f:
            h5_unwrapped = f['/trajectories/traj_001/coordinates/unwrapped'][:]
            h5_time = f['/trajectories/traj_001/time'][:]
            
            np.testing.assert_array_equal(
                sample_trajectory_data['unwrapped'],
                h5_unwrapped
            )
            np.testing.assert_array_equal(
                sample_trajectory_data['time'],
                h5_time
            )
```

## 7. Automated Testing Pipeline

### 7.1 Continuous Integration Setup

```yaml
# .github/workflows/trajectory_format_tests.yml
name: Trajectory Format Tests

on: [push, pull_request]

jobs:
  test-formats:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov h5py
    
    - name: Run format tests
      run: |
        pytest PyCD/tests/test_trajectory_formats.py -v
        pytest PyCD/tests/test_dual_writing.py -v
        pytest PyCD/tests/test_hdf5_schema.py -v
    
    - name: Run performance benchmarks
      run: |
        pytest PyCD/tests/test_performance_comparison.py -v --benchmark
    
    - name: Generate coverage report
      run: |
        pytest --cov=PyCD --cov-report=xml PyCD/tests/
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

### 7.2 Performance Regression Detection

```python
# performance_regression_test.py
class PerformanceRegression:
    """Detect performance regressions in trajectory I/O"""
    
    def __init__(self, baseline_file="performance_baseline.json"):
        self.baseline_file = baseline_file
        self.current_results = {}
        
    def record_benchmark(self, test_name, duration, file_size=None):
        """Record benchmark result"""
        self.current_results[test_name] = {
            'duration': duration,
            'file_size': file_size,
            'timestamp': time.time()
        }
    
    def check_regressions(self, tolerance=0.2):
        """Check for performance regressions"""
        if not os.path.exists(self.baseline_file):
            # First run - save baseline
            with open(self.baseline_file, 'w') as f:
                json.dump(self.current_results, f)
            return True
        
        with open(self.baseline_file, 'r') as f:
            baseline = json.load(f)
        
        regressions = []
        for test_name, current in self.current_results.items():
            if test_name in baseline:
                baseline_duration = baseline[test_name]['duration']
                current_duration = current['duration']
                
                regression_ratio = current_duration / baseline_duration
                if regression_ratio > (1 + tolerance):
                    regressions.append({
                        'test': test_name,
                        'baseline': baseline_duration,
                        'current': current_duration,
                        'regression': regression_ratio
                    })
        
        if regressions:
            print("Performance Regressions Detected:")
            for reg in regressions:
                print(f"  {reg['test']}: {reg['regression']:.1f}x slower")
            return False
        
        return True
```

This comprehensive testing strategy ensures robust validation of the HDF5 trajectory migration, covering data integrity, performance, error handling, and backward compatibility across the entire transition period.