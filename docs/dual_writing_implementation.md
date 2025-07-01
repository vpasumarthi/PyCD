# Dual-Writing Implementation Strategy

## Overview

This document outlines the technical approach for implementing dual-writing functionality that will simultaneously output trajectory data in both NumPy `.npy` and HDF5 formats during the transition period. This ensures backward compatibility while enabling users to gradually migrate to the new HDF5 format.

## 1. Architecture Design

### 1.1 Core Components

```python
# New classes to be added to PyCD/core.py

class TrajectoryWriterConfig:
    """Configuration for trajectory output formats"""
    def __init__(self, formats=['npy'], hdf5_options=None, validation=True):
        self.formats = formats  # ['npy', 'hdf5', 'both']
        self.hdf5_options = hdf5_options or {}
        self.validation = validation

class TrajectoryWriterManager:
    """Manages writing to multiple trajectory formats"""
    def __init__(self, config, metadata=None):
        self.config = config
        self.writers = {}
        self.metadata = metadata
        
    def initialize_writers(self, output_path, traj_id):
        """Initialize format-specific writers"""
        
    def write_frame(self, traj_id, frame_data):
        """Write frame to all configured formats"""
        
    def finalize(self):
        """Close all writers and perform validation"""

class HDF5TrajectoryWriter:
    """HDF5-specific trajectory writer"""
    def __init__(self, file_path, metadata, compression_opts):
        self.file_path = file_path
        self.metadata = metadata
        self.compression_opts = compression_opts
        self.h5file = None
        
    def create_trajectory_datasets(self, traj_id, estimated_frames):
        """Create HDF5 datasets for a trajectory"""
        
    def append_frame(self, traj_id, frame_data):
        """Append frame data to HDF5 datasets"""
        
    def finalize_trajectory(self, traj_id):
        """Finalize trajectory datasets"""

class NPYTrajectoryWriter:
    """NumPy .npy trajectory writer (existing functionality wrapped)"""
    def __init__(self, base_path):
        self.base_path = base_path
        
    def write_frame(self, traj_id, frame_data):
        """Write frame data using existing np.save approach"""
```

### 1.2 Integration Points

The dual-writing system will integrate with the existing codebase at these key points:

1. **Configuration Loading**: Extend YAML config parsing
2. **Run Initialization**: Set up trajectory writers
3. **Frame Writing**: Modify existing trajectory output loop
4. **Finalization**: Add cleanup and validation steps

## 2. Implementation Details

### 2.1 Configuration System

#### 2.1.1 YAML Configuration Extension
```yaml
# New section in sys_config.yml
trajectory_output:
  formats:
    npy: true                    # Continue writing .npy files
    hdf5: false                  # Enable HDF5 writing
  
  hdf5_options:
    compression: 'gzip'          # Compression algorithm
    compression_level: 6         # Compression level (0-9)
    chunking: true              # Enable chunking
    shuffle: true               # Improve compression
    fletcher32: false           # Checksum validation
    single_file: true           # All trajectories in one file vs separate files
    
  validation:
    cross_format_check: true    # Validate npy vs hdf5 equivalence
    tolerance: 1.0e-15          # Floating point comparison tolerance
    statistical_validation: true # Compare statistical properties
    
  performance:
    buffer_size: 1000           # Frames to buffer before writing
    parallel_writing: false     # Enable parallel I/O (future)
```

#### 2.1.2 Command Line Interface
```bash
# New command line options for trajectory scripts
python unwrap_trajectory.py input.npy --output-format hdf5 --output output.h5
python unwrap_trajectory.py input.npy --output-format both --output-dir ./results/

# For simulation runs
python -m PyCD.material_run --trajectory-format hdf5 --config config.yml
python -m PyCD.material_run --trajectory-format both --config config.yml
```

### 2.2 Core Implementation

#### 2.2.1 Modified Run.do_kmc_steps Method
```python
class Run:
    def __init__(self, ...):
        # Add trajectory writer manager
        self.traj_config = self._load_trajectory_config()
        self.traj_writer_manager = None
        
    def do_kmc_steps(self, dst_path, output_data, random_seed, compute_mode):
        # Initialize trajectory writer manager
        metadata = self._gather_trajectory_metadata()
        self.traj_writer_manager = TrajectoryWriterManager(
            self.traj_config, metadata
        )
        
        # Existing simulation loop with modifications
        for traj_index in range(self.n_traj):
            traj_id = f"traj_{traj_index+1:03d}"
            
            # Initialize writers for this trajectory
            traj_dir_path = dst_path.joinpath(f'traj{traj_index+1}')
            self.traj_writer_manager.initialize_writers(traj_dir_path, traj_id)
            
            # Existing KMC simulation loop
            while end_path_index < num_path_steps_per_traj:
                # ... existing KMC logic ...
                
                # Modified trajectory writing
                if should_write_frame:
                    frame_data = self._prepare_frame_data(
                        unwrapped_position_array, wrapped_position_array,
                        time_data, energy_array, delg_0_array, potential_array
                    )
                    self.traj_writer_manager.write_frame(traj_id, frame_data)
            
            # Finalize trajectory
            self.traj_writer_manager.finalize_trajectory(traj_id)
        
        # Final validation and cleanup
        if self.traj_config.validation:
            self.traj_writer_manager.validate_formats()
        self.traj_writer_manager.finalize()
```

#### 2.2.2 Frame Data Preparation
```python
def _prepare_frame_data(self, unwrapped_pos, wrapped_pos, time_data, 
                       energy, delg_0, potential):
    """Prepare frame data for writing to multiple formats"""
    
    frame_data = {}
    
    # Always include basic trajectory data
    if unwrapped_pos is not None:
        frame_data['coordinates/unwrapped'] = unwrapped_pos.copy()
    if wrapped_pos is not None:
        frame_data['coordinates/wrapped'] = wrapped_pos.copy()
    if time_data is not None:
        frame_data['time'] = time_data.copy()
    
    # Optional data based on output_data configuration
    if energy is not None and self.output_data['energy']['write']:
        frame_data['energy'] = energy.copy()
    if delg_0 is not None and self.output_data['delg_0']['write']:
        frame_data['delg_0'] = delg_0.copy()
    if potential is not None and self.output_data['potential']['write']:
        frame_data['potential'] = potential.copy()
    
    return frame_data
```

### 2.3 HDF5 Writer Implementation

#### 2.3.1 HDF5TrajectoryWriter Class
```python
import h5py
import numpy as np
from pathlib import Path

class HDF5TrajectoryWriter:
    def __init__(self, output_path, metadata, compression_opts):
        self.output_path = Path(output_path)
        self.metadata = metadata
        self.compression_opts = compression_opts
        self.h5file = None
        self.trajectories = {}
        self.frame_buffers = {}
        
    def initialize(self):
        """Initialize HDF5 file and metadata structure"""
        self.h5file = h5py.File(self.output_path, 'w')
        self._write_metadata()
        
    def _write_metadata(self):
        """Write simulation metadata to HDF5 file"""
        # Create metadata group structure
        meta = self.h5file.create_group('/metadata')
        
        # Simulation info
        sim_info = meta.create_group('simulation_info')
        sim_info.attrs['version'] = self.metadata['version']
        sim_info.attrs['created_on'] = self.metadata['created_on']
        sim_info.attrs['pycd_version'] = self.metadata['pycd_version']
        sim_info.attrs['format_version'] = 'hdf5_v1'
        
        # System info
        sys_info = meta.create_group('system_info')
        sys_info.create_dataset('lattice_matrix', 
                               data=self.metadata['lattice_matrix'])
        sys_info.create_dataset('pbc', data=self.metadata['pbc'])
        sys_info.create_dataset('system_size', 
                               data=self.metadata['system_size'])
        sys_info.attrs['temperature'] = self.metadata['temperature']
        
        # Species info
        species_info = sys_info.create_group('species_info')
        species_info.create_dataset('species_count', 
                                   data=self.metadata['species_count'])
        species_info.attrs['total_species'] = self.metadata['total_species']
        
        # Simulation parameters
        sim_params = meta.create_group('simulation_params')
        sim_params.attrs['time_interval'] = self.metadata['time_interval']
        sim_params.attrs['t_final'] = self.metadata['t_final']
        sim_params.attrs['n_traj'] = self.metadata['n_traj']
        sim_params.attrs['random_seed'] = self.metadata['random_seed']
        
    def create_trajectory_datasets(self, traj_id, estimated_frames=10000):
        """Create datasets for a new trajectory"""
        traj_group = self.h5file.create_group(f'/trajectories/{traj_id}')
        
        # Trajectory metadata
        traj_meta = traj_group.create_group('metadata')
        traj_meta.attrs['start_time'] = 0.0
        traj_meta.attrs['status'] = 'running'
        
        # Coordinate datasets
        coords_group = traj_group.create_group('coordinates')
        
        n_coords = self.metadata['total_species'] * 3
        chunk_size = min(1000, estimated_frames)
        
        # Unwrapped coordinates
        unwrapped = coords_group.create_dataset(
            'unwrapped',
            shape=(0, n_coords),
            maxshape=(None, n_coords),
            chunks=(chunk_size, n_coords),
            dtype=np.float64,
            **self.compression_opts
        )
        unwrapped.attrs['units'] = 'bohr'
        unwrapped.attrs['description'] = 'Unwrapped coordinates'
        unwrapped.attrs['species_layout'] = 'interleaved_xyz'
        
        # Wrapped coordinates
        wrapped = coords_group.create_dataset(
            'wrapped',
            shape=(0, n_coords),
            maxshape=(None, n_coords),
            chunks=(chunk_size, n_coords),
            dtype=np.float64,
            **self.compression_opts
        )
        wrapped.attrs['units'] = 'bohr'
        wrapped.attrs['description'] = 'PBC-wrapped coordinates'
        
        # Time dataset
        time_ds = traj_group.create_dataset(
            'time',
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_size,),
            dtype=np.float64,
            **self.compression_opts
        )
        time_ds.attrs['units'] = 'seconds'
        time_ds.attrs['description'] = 'Simulation time'
        
        # Energy dataset (optional)
        energy_ds = traj_group.create_dataset(
            'energy',
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_size,),
            dtype=np.float64,
            **self.compression_opts
        )
        energy_ds.attrs['units'] = 'hartree'
        energy_ds.attrs['description'] = 'Total system energy'
        
        self.trajectories[traj_id] = traj_group
        self.frame_buffers[traj_id] = {
            'unwrapped': [],
            'wrapped': [],
            'time': [],
            'energy': [],
            'frame_count': 0
        }
        
    def append_frame(self, traj_id, frame_data):
        """Append frame data to trajectory"""
        if traj_id not in self.trajectories:
            raise ValueError(f"Trajectory {traj_id} not initialized")
            
        buffer = self.frame_buffers[traj_id]
        
        # Add frame data to buffers
        if 'coordinates/unwrapped' in frame_data:
            buffer['unwrapped'].append(frame_data['coordinates/unwrapped'])
        if 'coordinates/wrapped' in frame_data:
            buffer['wrapped'].append(frame_data['coordinates/wrapped'])
        if 'time' in frame_data:
            buffer['time'].append(frame_data['time'])
        if 'energy' in frame_data:
            buffer['energy'].append(frame_data['energy'])
            
        buffer['frame_count'] += 1
        
        # Flush buffer if it's full
        if buffer['frame_count'] >= 1000:  # Buffer size
            self._flush_buffer(traj_id)
            
    def _flush_buffer(self, traj_id):
        """Flush buffered data to HDF5 datasets"""
        buffer = self.frame_buffers[traj_id]
        traj_group = self.trajectories[traj_id]
        
        if buffer['frame_count'] == 0:
            return
            
        # Resize datasets and write buffered data
        for data_type in ['unwrapped', 'wrapped']:
            if buffer[data_type]:
                dataset_path = f'coordinates/{data_type}'
                if dataset_path in traj_group:
                    dataset = traj_group[dataset_path]
                    old_size = dataset.shape[0]
                    new_size = old_size + len(buffer[data_type])
                    dataset.resize((new_size, dataset.shape[1]))
                    dataset[old_size:new_size] = np.array(buffer[data_type])
                    buffer[data_type] = []
        
        # Time and energy data
        for data_type in ['time', 'energy']:
            if buffer[data_type] and data_type in traj_group:
                dataset = traj_group[data_type]
                old_size = dataset.shape[0]
                new_size = old_size + len(buffer[data_type])
                dataset.resize((new_size,))
                dataset[old_size:new_size] = np.array(buffer[data_type])
                buffer[data_type] = []
        
        buffer['frame_count'] = 0
        
    def finalize_trajectory(self, traj_id):
        """Finalize trajectory and flush remaining data"""
        self._flush_buffer(traj_id)
        
        # Update trajectory metadata
        traj_group = self.trajectories[traj_id]
        meta = traj_group['metadata']
        meta.attrs['status'] = 'completed'
        meta.attrs['n_frames'] = traj_group['time'].shape[0]
        
        if 'time' in traj_group and traj_group['time'].shape[0] > 0:
            meta.attrs['end_time'] = traj_group['time'][-1]
            
    def finalize(self):
        """Close HDF5 file"""
        if self.h5file:
            self.h5file.close()
```

### 2.4 Cross-Format Validation

#### 2.4.1 Validation Framework
```python
class TrajectoryValidator:
    """Validate equivalence between .npy and HDF5 trajectory data"""
    
    def __init__(self, tolerance=1e-15):
        self.tolerance = tolerance
        self.validation_results = {}
        
    def validate_trajectory(self, npy_dir, hdf5_file, traj_id):
        """Compare trajectory data between formats"""
        results = {
            'coordinates_match': False,
            'time_match': False,
            'energy_match': False,
            'errors': []
        }
        
        try:
            # Load NPY data
            npy_data = self._load_npy_trajectory(npy_dir)
            
            # Load HDF5 data
            h5_data = self._load_hdf5_trajectory(hdf5_file, traj_id)
            
            # Compare coordinates
            if 'unwrapped_coords' in npy_data and 'unwrapped_coords' in h5_data:
                coord_match = np.allclose(
                    npy_data['unwrapped_coords'],
                    h5_data['unwrapped_coords'],
                    atol=self.tolerance
                )
                results['coordinates_match'] = coord_match
                if not coord_match:
                    max_diff = np.max(np.abs(
                        npy_data['unwrapped_coords'] - h5_data['unwrapped_coords']
                    ))
                    results['errors'].append(f"Coordinate mismatch: max_diff={max_diff}")
            
            # Compare time data
            if 'time' in npy_data and 'time' in h5_data:
                time_match = np.allclose(
                    npy_data['time'],
                    h5_data['time'],
                    atol=self.tolerance
                )
                results['time_match'] = time_match
                if not time_match:
                    max_diff = np.max(np.abs(npy_data['time'] - h5_data['time']))
                    results['errors'].append(f"Time data mismatch: max_diff={max_diff}")
            
            # Compare energy data
            if 'energy' in npy_data and 'energy' in h5_data:
                energy_match = np.allclose(
                    npy_data['energy'],
                    h5_data['energy'],
                    atol=self.tolerance
                )
                results['energy_match'] = energy_match
                if not energy_match:
                    max_diff = np.max(np.abs(npy_data['energy'] - h5_data['energy']))
                    results['errors'].append(f"Energy data mismatch: max_diff={max_diff}")
                    
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            
        self.validation_results[traj_id] = results
        return results
        
    def _load_npy_trajectory(self, npy_dir):
        """Load trajectory data from .npy files"""
        data = {}
        npy_path = Path(npy_dir)
        
        files_to_load = [
            ('unwrapped_traj.npy', 'unwrapped_coords'),
            ('wrapped_traj.npy', 'wrapped_coords'),
            ('time.npy', 'time'),
            ('energy.npy', 'energy')
        ]
        
        for filename, key in files_to_load:
            filepath = npy_path / filename
            if filepath.exists():
                data[key] = np.load(filepath)
                
        return data
        
    def _load_hdf5_trajectory(self, hdf5_file, traj_id):
        """Load trajectory data from HDF5 file"""
        data = {}
        
        with h5py.File(hdf5_file, 'r') as f:
            traj_group = f[f'/trajectories/{traj_id}']
            
            # Load coordinate data
            if 'coordinates/unwrapped' in traj_group:
                data['unwrapped_coords'] = traj_group['coordinates/unwrapped'][:]
            if 'coordinates/wrapped' in traj_group:
                data['wrapped_coords'] = traj_group['coordinates/wrapped'][:]
                
            # Load time and energy data
            if 'time' in traj_group:
                data['time'] = traj_group['time'][:]
            if 'energy' in traj_group:
                data['energy'] = traj_group['energy'][:]
                
        return data
```

## 3. Testing Strategy

### 3.1 Unit Tests
```python
# New test file: PyCD/tests/test_dual_writing.py

def test_hdf5_writer_initialization():
    """Test HDF5 writer creates proper file structure"""
    
def test_dual_writing_equivalence():
    """Test that .npy and HDF5 contain identical data"""
    
def test_incremental_writing():
    """Test that data can be written incrementally"""
    
def test_compression_integrity():
    """Test that compressed data matches uncompressed"""
    
def test_trajectory_validation():
    """Test cross-format validation functionality"""
    
def test_configuration_loading():
    """Test trajectory configuration loading"""
```

### 3.2 Integration Tests
```python
def test_full_simulation_dual_writing():
    """Test complete simulation with dual writing enabled"""
    
def test_performance_overhead():
    """Measure performance impact of dual writing"""
    
def test_storage_efficiency():
    """Compare storage requirements between formats"""
```

## 4. Performance Optimization

### 4.1 Buffering Strategy
- Buffer frames in memory before writing to reduce I/O overhead
- Configurable buffer size based on available memory
- Efficient array concatenation and resizing

### 4.2 Parallel I/O (Future Enhancement)
```python
# Future implementation for parallel writing
class ParallelTrajectoryWriter:
    def __init__(self, config):
        self.config = config
        self.write_queue = Queue()
        self.worker_threads = []
        
    def start_workers(self):
        """Start background threads for writing"""
        
    def queue_frame(self, traj_id, frame_data):
        """Queue frame for background writing"""
```

## 5. Error Handling and Recovery

### 5.1 Write Failure Recovery
```python
def handle_write_failure(self, error, traj_id, frame_data):
    """Handle write failures gracefully"""
    
    # Log error with context
    logger.error(f"Write failure for {traj_id}: {error}")
    
    # Attempt to save frame data to backup location
    backup_path = self.create_backup_location(traj_id)
    self.save_frame_backup(backup_path, frame_data)
    
    # Continue with remaining format writers
    if self.config.continue_on_error:
        return True
    else:
        raise error
```

### 5.2 Partial Data Recovery
```python
def recover_partial_trajectory(self, hdf5_file, traj_id):
    """Recover trajectory from partial HDF5 data"""
    
    with h5py.File(hdf5_file, 'r+') as f:
        traj_group = f[f'/trajectories/{traj_id}']
        
        # Mark trajectory as incomplete
        traj_group['metadata'].attrs['status'] = 'incomplete'
        
        # Truncate datasets to last complete frame
        last_complete_frame = self.find_last_complete_frame(traj_group)
        self.truncate_datasets(traj_group, last_complete_frame)
```

This dual-writing implementation strategy provides a robust framework for transitioning to HDF5 while maintaining backward compatibility and ensuring data integrity throughout the migration process.