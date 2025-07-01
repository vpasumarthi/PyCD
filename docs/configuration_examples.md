# Example Configuration for HDF5 Dual-Writing

## Configuration File Example

```yaml
# sys_config.yml - Extended configuration for HDF5 trajectory support

# Existing PyCD configuration...
material_parameters:
  # ... existing material parameters ...

# NEW: Trajectory output configuration
trajectory_output:
  # Format selection - can be 'npy', 'hdf5', or 'both'
  formats:
    npy: true                    # Continue writing .npy files (backward compatibility)
    hdf5: false                  # Enable HDF5 writing (opt-in initially)
  
  # HDF5-specific options
  hdf5_options:
    # File organization
    single_file: true            # All trajectories in one file vs separate files
    filename_pattern: "trajectory_{timestamp}.h5"  # Custom filename pattern
    
    # Performance settings
    compression: 'gzip'          # Compression algorithm ('gzip', 'lzf', 'szip', or null)
    compression_level: 6         # Compression level (0-9, higher = better compression)
    chunking: true              # Enable chunking for append operations
    chunk_size: 1000            # Frames per chunk (auto-calculated if null)
    shuffle: true               # Improve compression ratio
    fletcher32: false           # Enable checksums (slight performance cost)
    
    # Buffer settings
    buffer_frames: 100          # Frames to buffer before writing
    flush_frequency: 1000       # Flush buffer every N frames
    
    # Metadata options
    include_simulation_metadata: true    # Include full simulation parameters
    include_analysis_metadata: false    # Include analysis results (future)
    custom_attributes: {}               # User-defined attributes
    
  # Cross-format validation
  validation:
    cross_format_check: true    # Validate equivalence between .npy and HDF5
    tolerance: 1.0e-15          # Floating point comparison tolerance
    statistical_validation: true # Compare statistical properties
    validation_frequency: 'end' # When to validate: 'frame', 'chunk', 'end'
    
  # Error handling
  error_handling:
    continue_on_write_error: true    # Continue simulation if one format fails
    backup_failed_frames: true      # Save failed frames to backup location
    max_retry_attempts: 3           # Retry failed writes
    
  # Performance monitoring
  performance:
    track_write_times: true         # Monitor write performance
    track_file_sizes: true          # Monitor storage efficiency
    performance_report: true        # Generate performance summary
```

## Command Line Usage Examples

```bash
# Use existing .npy format (default, unchanged)
python -m PyCD.material_run --config sys_config.yml

# Enable HDF5 format only
python -m PyCD.material_run --config sys_config.yml --trajectory-format hdf5

# Enable dual-writing (both formats)
python -m PyCD.material_run --config sys_config.yml --trajectory-format both

# Override compression settings
python -m PyCD.material_run --config sys_config.yml \
    --trajectory-format hdf5 \
    --hdf5-compression gzip \
    --hdf5-compression-level 9

# Disable validation for performance
python -m PyCD.material_run --config sys_config.yml \
    --trajectory-format both \
    --no-validation
```

## Trajectory Analysis Examples

```bash
# Analyze existing .npy trajectory (unchanged)
python -m PyCD.scripts.unwrap_trajectory wrapped_traj.npy \
    --system-size 2 2 1 \
    --lattice-diagonal 10.0 10.0 12.0 \
    -o unwrapped_traj.npy

# Analyze HDF5 trajectory
python -m PyCD.scripts.unwrap_trajectory trajectory.h5 \
    --trajectory-id traj_001 \
    --system-size 2 2 1 \
    --output-format hdf5 \
    -o unwrapped_trajectory.h5

# Convert between formats
python -m PyCD.scripts.convert_trajectory \
    --input wrapped_traj.npy \
    --output trajectory.h5 \
    --input-format npy \
    --output-format hdf5 \
    --metadata-file simulation_metadata.yml
```

## Python API Examples

```python
# Example 1: Reading HDF5 trajectory data
import h5py
import numpy as np

def read_hdf5_trajectory(filename, trajectory_id='traj_001'):
    """Read trajectory data from HDF5 file"""
    
    with h5py.File(filename, 'r') as f:
        # Read metadata
        metadata = {}
        meta_group = f['/metadata']
        for group_name in meta_group.keys():
            metadata[group_name] = {}
            group = meta_group[group_name]
            
            # Read attributes
            for attr_name in group.attrs.keys():
                metadata[group_name][attr_name] = group.attrs[attr_name]
            
            # Read datasets
            for dataset_name in group.keys():
                metadata[group_name][dataset_name] = group[dataset_name][:]
        
        # Read trajectory data
        traj_group = f[f'/trajectories/{trajectory_id}']
        trajectory_data = {}
        
        # Coordinates
        coords = traj_group['coordinates']
        trajectory_data['unwrapped'] = coords['unwrapped'][:]
        trajectory_data['wrapped'] = coords['wrapped'][:]
        
        # Time and energy data
        trajectory_data['time'] = traj_group['time'][:]
        trajectory_data['energy'] = traj_group['energy'][:]
        
        # Optional data
        if 'delg_0' in traj_group:
            trajectory_data['delg_0'] = traj_group['delg_0'][:]
        if 'occupancy' in traj_group:
            trajectory_data['occupancy'] = traj_group['occupancy'][:]
    
    return metadata, trajectory_data

# Usage
metadata, traj_data = read_hdf5_trajectory('simulation_results.h5', 'traj_001')
print(f"Simulation temperature: {metadata['system_info']['temperature']} K")
print(f"Trajectory shape: {traj_data['unwrapped'].shape}")
```

```python
# Example 2: Configuring dual-writing programmatically
from PyCD.core import TrajectoryWriterConfig, TrajectoryWriterManager

def setup_dual_writing():
    """Configure dual-writing for trajectory output"""
    
    # Configure formats and options
    config = TrajectoryWriterConfig(
        formats=['npy', 'hdf5'],
        hdf5_options={
            'compression': 'gzip',
            'compression_opts': 6,
            'chunks': True,
            'shuffle': True,
            'single_file': True
        },
        validation={
            'cross_format_check': True,
            'tolerance': 1e-15
        }
    )
    
    # Simulation metadata
    metadata = {
        'version': '1.0.0',
        'created_on': '2024-01-01T00:00:00Z',
        'lattice_matrix': np.eye(3) * 10.0,
        'pbc': np.array([1, 1, 1]),
        'system_size': np.array([2, 2, 1]),
        'species_count': np.array([5, 5]),
        'total_species': 10,
        'temperature': 300.0,
        'time_interval': 1e-12,
        't_final': 1e-9,
        'n_traj': 5
    }
    
    return config, metadata

# Usage in simulation code
config, metadata = setup_dual_writing()
writer_manager = TrajectoryWriterManager(config, metadata)

# In simulation loop
for traj_id in range(n_trajectories):
    writer_manager.initialize_writers(output_path, f"traj_{traj_id:03d}")
    
    for frame in simulation_frames:
        frame_data = {
            'coordinates/unwrapped': unwrapped_coords,
            'coordinates/wrapped': wrapped_coords,
            'time': simulation_time,
            'energy': system_energy
        }
        writer_manager.write_frame(f"traj_{traj_id:03d}", frame_data)
    
    writer_manager.finalize_trajectory(f"traj_{traj_id:03d}")

# Final validation and cleanup
writer_manager.validate_formats()
writer_manager.finalize()
```

```python
# Example 3: Migration utility
def migrate_npy_to_hdf5(npy_directory, hdf5_filename, metadata=None):
    """Migrate existing .npy trajectory files to HDF5 format"""
    
    import os
    from pathlib import Path
    
    npy_dir = Path(npy_directory)
    
    # Discover .npy files
    npy_files = {
        'unwrapped': npy_dir / 'unwrapped_traj.npy',
        'wrapped': npy_dir / 'wrapped_traj.npy', 
        'time': npy_dir / 'time.npy',
        'energy': npy_dir / 'energy.npy',
        'delg_0': npy_dir / 'delg_0.npy',
        'occupancy': npy_dir / 'occupancy.npy'
    }
    
    # Load data that exists
    trajectory_data = {}
    for data_type, filepath in npy_files.items():
        if filepath.exists():
            trajectory_data[data_type] = np.load(filepath)
            print(f"Loaded {data_type}: {trajectory_data[data_type].shape}")
    
    # Create HDF5 file
    with h5py.File(hdf5_filename, 'w') as f:
        # Write metadata if provided
        if metadata:
            meta_group = f.create_group('/metadata')
            # ... write metadata structure ...
        
        # Create trajectory group
        traj_group = f.create_group('/trajectories/traj_001')
        coords_group = traj_group.create_group('coordinates')
        
        # Write coordinate data
        if 'unwrapped' in trajectory_data:
            coords_group.create_dataset(
                'unwrapped',
                data=trajectory_data['unwrapped'],
                compression='gzip',
                compression_opts=6,
                chunks=True
            )
        
        if 'wrapped' in trajectory_data:
            coords_group.create_dataset(
                'wrapped', 
                data=trajectory_data['wrapped'],
                compression='gzip',
                compression_opts=6,
                chunks=True
            )
        
        # Write time and energy data
        for data_type in ['time', 'energy', 'delg_0']:
            if data_type in trajectory_data:
                traj_group.create_dataset(
                    data_type,
                    data=trajectory_data[data_type],
                    compression='gzip',
                    compression_opts=6,
                    chunks=True
                )
        
        # Write occupancy data if present
        if 'occupancy' in trajectory_data:
            traj_group.create_dataset(
                'occupancy',
                data=trajectory_data['occupancy'],
                compression='gzip',
                compression_opts=6,
                chunks=True,
                dtype=np.int32
            )
    
    print(f"Migration complete: {hdf5_filename}")
    
    # Validate migration
    validation_results = validate_migration(npy_directory, hdf5_filename)
    return validation_results

def validate_migration(npy_directory, hdf5_filename):
    """Validate that migration preserved data integrity"""
    
    results = {'status': 'success', 'errors': []}
    
    # Load original .npy data
    npy_dir = Path(npy_directory)
    npy_unwrapped = np.load(npy_dir / 'unwrapped_traj.npy')
    npy_time = np.load(npy_dir / 'time.npy')
    
    # Load HDF5 data
    with h5py.File(hdf5_filename, 'r') as f:
        h5_unwrapped = f['/trajectories/traj_001/coordinates/unwrapped'][:]
        h5_time = f['/trajectories/traj_001/time'][:]
    
    # Compare data
    if not np.array_equal(npy_unwrapped, h5_unwrapped):
        results['status'] = 'failed'
        results['errors'].append('Unwrapped coordinates do not match')
    
    if not np.array_equal(npy_time, h5_time):
        results['status'] = 'failed'
        results['errors'].append('Time data does not match')
    
    if results['status'] == 'success':
        print("✅ Migration validation successful")
    else:
        print("❌ Migration validation failed:")
        for error in results['errors']:
            print(f"  - {error}")
    
    return results

# Usage
metadata = {
    'version': '1.0.0',
    'temperature': 300.0,
    'lattice_matrix': np.eye(3) * 10.0,
    # ... other metadata ...
}

validation = migrate_npy_to_hdf5('./traj1/', 'migrated_trajectory.h5', metadata)
```

## File Organization Examples

```
# Current .npy organization (unchanged)
SimulationFiles/
├── traj1/
│   ├── unwrapped_traj.npy
│   ├── wrapped_traj.npy
│   ├── time.npy
│   ├── energy.npy
│   └── delg_0.npy
├── traj2/
│   └── ...
└── ...

# Option 1: Single HDF5 file (recommended)
SimulationFiles/
├── trajectory_data.h5          # All trajectories in one file
├── traj1/                      # Legacy .npy files (during dual-writing)
│   ├── unwrapped_traj.npy
│   └── ...
└── ...

# Option 2: Per-trajectory HDF5 files
SimulationFiles/
├── traj1/
│   ├── trajectory.h5           # HDF5 format
│   ├── unwrapped_traj.npy     # Legacy .npy files
│   └── ...
├── traj2/
│   ├── trajectory.h5
│   └── ...
└── ...

# Option 3: Separate HDF5 directory
SimulationFiles/
├── npy_format/                 # Legacy .npy files
│   ├── traj1/
│   └── traj2/
├── hdf5_format/               # HDF5 files
│   ├── trajectory_data.h5
│   └── metadata.yml
└── analysis_results/          # Analysis outputs
```

This configuration system provides flexible, backward-compatible trajectory storage options while enabling users to gradually adopt HDF5 format benefits.