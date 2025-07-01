# HDF5 Trajectory Schema Specification

## Version: 1.0
## Date: 2024-12-19

## 1. File Format Specification

### 1.1 File Structure Overview
```
trajectory.h5
├── /metadata                    # Root metadata group
├── /trajectories               # Trajectory data group  
└── /analysis                   # Optional analysis results group
```

### 1.2 Detailed Schema

#### 1.2.1 Metadata Group (`/metadata`)
```
/metadata
├── simulation_info (Group)
│   ├── version (Attribute): "1.0.0"
│   ├── created_on (Attribute): "2024-01-01T00:00:00Z" 
│   ├── pycd_version (Attribute): "1.0.0"
│   ├── format_version (Attribute): "hdf5_v1"
│   ├── total_trajectories (Attribute): int
│   ├── description (Attribute): str (optional)
│   └── notes (Attribute): str (optional)
├── system_info (Group)
│   ├── lattice_matrix (Dataset): float64[3,3]
│   ├── pbc (Dataset): int32[3] 
│   ├── system_size (Dataset): int32[3]
│   ├── species_info (Group)
│   │   ├── species_count (Dataset): int32[n_species]
│   │   ├── species_names (Dataset): string[n_species]
│   │   ├── species_charges (Dataset): float64[n_species]
│   │   └── total_species (Attribute): int
│   ├── temperature (Attribute): float64
│   └── material_properties (Group, optional)
│       ├── dielectric_constant (Attribute): float64
│       ├── material_name (Attribute): str
│       └── crystal_system (Attribute): str
└── simulation_params (Group)
    ├── time_interval (Attribute): float64
    ├── t_final (Attribute): float64  
    ├── n_traj (Attribute): int
    ├── random_seed (Attribute): int
    ├── kmc_settings (Group, optional)
    │   ├── max_steps (Attribute): int
    │   ├── energy_cutoff (Attribute): float64
    │   └── hop_distance_cutoff (Attribute): float64
    └── field_settings (Group, optional)
        ├── electric_field (Dataset): float64[3]
        ├── field_strength (Attribute): float64
        └── field_direction (Dataset): float64[3]
```

#### 1.2.2 Trajectories Group (`/trajectories`)
```
/trajectories
├── traj_001 (Group)
│   ├── metadata (Group)
│   │   ├── n_frames (Attribute): int
│   │   ├── start_time (Attribute): float64
│   │   ├── end_time (Attribute): float64
│   │   ├── random_seed (Attribute): int
│   │   └── status (Attribute): "completed"|"interrupted"|"error"
│   ├── coordinates (Group)
│   │   ├── unwrapped (Dataset): float64[n_frames, n_species*3]
│   │   │   ├── units (Attribute): "bohr"|"angstrom" 
│   │   │   ├── description (Attribute): "Unwrapped coordinates"
│   │   │   └── species_layout (Attribute): "interleaved_xyz"
│   │   └── wrapped (Dataset): float64[n_frames, n_species*3]
│   │       ├── units (Attribute): "bohr"|"angstrom"
│   │       ├── description (Attribute): "PBC-wrapped coordinates"
│   │       └── species_layout (Attribute): "interleaved_xyz"
│   ├── time (Dataset): float64[n_frames]
│   │   ├── units (Attribute): "seconds"
│   │   └── description (Attribute): "Simulation time"
│   ├── energy (Dataset): float64[n_frames]
│   │   ├── units (Attribute): "hartree"|"eV"
│   │   └── description (Attribute): "Total system energy"
│   ├── delg_0 (Dataset, optional): float64[n_frames]
│   │   ├── units (Attribute): "hartree"|"eV"
│   │   └── description (Attribute): "Free energy change"
│   ├── potential (Dataset, optional): float64[n_frames]
│   │   ├── units (Attribute): "hartree"|"eV"
│   │   └── description (Attribute): "Potential energy"
│   └── occupancy (Dataset, optional): int32[n_frames, n_sites]
│       ├── description (Attribute): "Site occupancy states"
│       └── encoding (Attribute): "0=empty, 1=electron, 2=hole"
├── traj_002 (Group)
│   └── ... (same structure as traj_001)
└── ...
```

#### 1.2.3 Analysis Group (`/analysis`, optional)
```
/analysis
├── msd (Group, optional)
│   ├── time_points (Dataset): float64[n_points]
│   ├── msd_values (Dataset): float64[n_points, n_species]
│   ├── diffusion_coefficients (Dataset): float64[n_species]
│   └── analysis_params (Group)
│       ├── trim_length (Attribute): float64
│       ├── temperature (Attribute): float64
│       └── method (Attribute): str
└── conductivity (Group, optional)
    ├── sigma_values (Dataset): float64[n_trajectories]
    ├── mobility_values (Dataset): float64[n_species]
    └── analysis_params (Group)
        ├── field_strength (Attribute): float64
        └── temperature (Attribute): float64
```

## 2. Data Type Specifications

### 2.1 Coordinate Data Layout
```python
# Species layout: interleaved XYZ coordinates
# For n_species=2, n_frames=1000:
coordinates.shape = (1000, 6)  # [x1,y1,z1,x2,y2,z2] per frame

# Alternative: grouped by species (future consideration)
# coordinates.shape = (1000, 2, 3)  # [frame][species][xyz]
```

### 2.2 Dataset Properties
```python
# Standard dataset creation parameters
dataset_params = {
    'chunks': True,           # Enable chunking for append operations
    'compression': 'gzip',    # Use gzip compression
    'compression_opts': 6,    # Compression level (0-9)
    'shuffle': True,          # Improve compression ratio
    'fletcher32': False,      # Checksum (optional)
    'maxshape': (None, ...),  # Allow unlimited growth in time dimension
}
```

### 2.3 Units and Conventions
- **Length**: Bohr radii (default) or Angstroms (with units attribute)
- **Time**: Seconds (SI units)
- **Energy**: Hartree (default) or eV (with units attribute)
- **Temperature**: Kelvin
- **Coordinates**: Cartesian coordinates in simulation box frame

## 3. Schema Validation

### 3.1 Required Attributes
```python
required_attributes = {
    '/metadata/simulation_info': ['version', 'created_on', 'format_version'],
    '/metadata/system_info': ['temperature'],
    '/metadata/simulation_params': ['time_interval', 't_final', 'n_traj'],
}

required_datasets = {
    '/metadata/system_info': ['lattice_matrix', 'pbc', 'system_size'],
    '/trajectories/traj_XXX/coordinates': ['unwrapped'],
    '/trajectories/traj_XXX': ['time'],
}
```

### 3.2 Data Validation Rules
```python
validation_rules = {
    'lattice_matrix': {
        'shape': (3, 3),
        'dtype': 'float64',
        'constraints': 'positive_definite'
    },
    'pbc': {
        'shape': (3,),
        'dtype': 'int32', 
        'values': [0, 1]
    },
    'coordinates/unwrapped': {
        'shape': ('n_frames', 'n_species*3'),
        'dtype': 'float64',
        'constraints': 'finite_values'
    },
    'time': {
        'shape': ('n_frames',),
        'dtype': 'float64',
        'constraints': 'monotonic_increasing'
    }
}
```

## 4. Backward Compatibility

### 4.1 Format Version Evolution
```python
format_versions = {
    'hdf5_v1': {
        'version': '1.0',
        'compatible_with': [],
        'breaking_changes': 'Initial HDF5 format'
    },
    'hdf5_v2': {  # Future version
        'version': '2.0', 
        'compatible_with': ['hdf5_v1'],
        'breaking_changes': 'Added new coordinate layout option'
    }
}
```

### 4.2 Migration Support
```python
# Conversion utilities to be implemented
def convert_npy_to_hdf5(npy_directory, hdf5_file, metadata):
    """Convert existing .npy trajectory to HDF5 format"""
    
def convert_hdf5_to_npy(hdf5_file, output_directory):
    """Convert HDF5 trajectory back to .npy format"""
    
def validate_hdf5_schema(hdf5_file):
    """Validate HDF5 file against schema specification"""
```

## 5. Performance Considerations

### 5.1 Chunking Strategy
```python
# Recommended chunk sizes for different trajectory lengths
chunk_sizes = {
    'short_traj': {     # < 10,000 frames
        'time_chunks': (1000,),
        'coord_chunks': (1000, 'auto'),
    },
    'medium_traj': {    # 10,000 - 100,000 frames  
        'time_chunks': (5000,),
        'coord_chunks': (5000, 'auto'),
    },
    'long_traj': {      # > 100,000 frames
        'time_chunks': (10000,),
        'coord_chunks': (10000, 'auto'),
    }
}
```

### 5.2 Compression Benchmarks
```python
# Expected compression ratios (approximate)
compression_ratios = {
    'coordinates': {
        'uncompressed': 1.0,
        'gzip_level_1': 2.5,
        'gzip_level_6': 3.2,  # Recommended
        'gzip_level_9': 3.4,
    },
    'time': {
        'uncompressed': 1.0,
        'gzip_level_6': 4.1,  # Highly compressible
    },
    'energy': {
        'uncompressed': 1.0,
        'gzip_level_6': 2.8,
    }
}
```

## 6. Example Usage

### 6.1 Reading HDF5 Trajectory
```python
import h5py
import numpy as np

with h5py.File('trajectory.h5', 'r') as f:
    # Read metadata
    lattice = f['/metadata/system_info/lattice_matrix'][:]
    pbc = f['/metadata/system_info/pbc'][:]
    
    # Read first trajectory
    traj1 = f['/trajectories/traj_001']
    unwrapped_coords = traj1['coordinates/unwrapped'][:]
    time_data = traj1['time'][:]
    
    # Read specific frame range
    frames_100_200 = traj1['coordinates/unwrapped'][100:200]
```

### 6.2 Writing HDF5 Trajectory
```python
import h5py
import numpy as np

with h5py.File('trajectory.h5', 'w') as f:
    # Create metadata
    meta = f.create_group('/metadata')
    sim_info = meta.create_group('simulation_info')
    sim_info.attrs['version'] = '1.0.0'
    sim_info.attrs['format_version'] = 'hdf5_v1'
    
    # Create trajectory group
    traj = f.create_group('/trajectories/traj_001')
    
    # Create datasets with chunking and compression
    coords = traj.create_dataset(
        'coordinates/unwrapped',
        shape=(0, 6),  # Start empty, will append
        maxshape=(None, 6),
        chunks=(1000, 6),
        compression='gzip',
        compression_opts=6,
        dtype=np.float64
    )
    
    # Append data as simulation progresses
    new_frame = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    coords.resize((coords.shape[0] + 1, coords.shape[1]))
    coords[-1:] = new_frame
```

## 7. Interoperability Considerations

### 7.1 External Tool Compatibility
- **MDAnalysis**: Compatible trajectory format for analysis
- **VMD**: Can read HDF5 with custom plugins
- **ASE**: Compatible with Atoms objects
- **OVITO**: Can be extended to read HDF5 trajectories

### 7.2 Standard Compliance
- Follow HDF5 best practices for scientific data
- Use standard attribute names where possible
- Provide clear documentation for custom extensions
- Include version information for schema evolution

This schema specification provides a comprehensive framework for storing trajectory data in HDF5 format while maintaining flexibility for future enhancements and ensuring interoperability with external tools.