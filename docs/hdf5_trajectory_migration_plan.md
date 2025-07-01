# HDF5 Trajectory Format Migration Plan

## Executive Summary

This document outlines a comprehensive plan for transitioning PyCD's trajectory storage from NumPy `.npy` format to HDF5, implementing a dual-writing strategy during the transition phase to ensure backward compatibility and data integrity.

## 1. Current State Analysis

### 1.1 Existing Trajectory Storage
- **Format**: NumPy binary `.npy` files
- **Location**: `PyCD/core.py`, `Run.do_kmc_steps()` method (lines 2842-2863)
- **Data Types**:
  - `unwrapped_traj`: Continuous particle coordinates
  - `wrapped_traj`: Periodic boundary condition-wrapped coordinates  
  - `time`: Simulation time stamps
  - `energy`: System energy at each frame
  - `delg_0`: Free energy change data
  - `potential`: Potential energy data
  - `occupancy`: Dopant occupancy states (when doping is active)

### 1.2 Current File Organization
```
SimulationFiles/
├── traj1/
│   ├── unwrapped_traj.npy
│   ├── wrapped_traj.npy
│   ├── time.npy
│   ├── energy.npy
│   └── ...
├── traj2/
│   └── ...
```

### 1.3 Current I/O Implementation
- **Writing**: `np.save()` with append mode (`'ab'`)
- **Reading**: `np.load()` for analysis and post-processing
- **Incremental Writing**: Trajectory chunks appended during simulation

## 2. HDF5 Schema Design

### 2.1 Proposed HDF5 File Structure
```
trajectory.h5
├── /metadata
│   ├── simulation_info (group)
│   │   ├── version: "1.0.0"
│   │   ├── created_on: "2024-01-01T00:00:00Z"
│   │   ├── pycd_version: "1.0.0"
│   │   ├── total_trajectories: 10
│   │   └── format_version: "hdf5_v1"
│   ├── system_info (group)
│   │   ├── lattice_matrix: [[a,b,c], [d,e,f], [g,h,i]]
│   │   ├── pbc: [1,1,1]
│   │   ├── system_size: [nx, ny, nz]
│   │   ├── species_count: [n_electrons, n_holes]
│   │   ├── total_species: n_total
│   │   └── temperature: 300.0
│   └── simulation_params (group)
│       ├── time_interval: 1e-12
│       ├── t_final: 1e-9
│       ├── n_traj: 10
│       └── random_seed: 12345
├── /trajectories
│   ├── traj_001 (group)
│   │   ├── coordinates (group)
│   │   │   ├── unwrapped: [n_frames, n_species*3] (chunked, compressed)
│   │   │   └── wrapped: [n_frames, n_species*3] (chunked, compressed)
│   │   ├── time: [n_frames] (chunked, compressed)
│   │   ├── energy: [n_frames] (chunked, compressed)
│   │   ├── delg_0: [n_frames] (chunked, compressed)
│   │   ├── potential: [n_frames] (chunked, compressed)
│   │   └── occupancy: [n_frames, n_sites] (chunked, compressed, optional)
│   ├── traj_002 (group)
│   │   └── ...
│   └── ...
```

### 2.2 Dataset Properties
- **Chunking**: Enable efficient append operations and partial reading
- **Compression**: Use gzip or lzf for space efficiency
- **Data Types**: 
  - Float64 for coordinates, energy, time
  - Int32 for occupancy states
- **Attributes**: Frame-level metadata when needed

### 2.3 Metadata Schema
```python
# Simulation-level metadata
simulation_info = {
    'version': '1.0.0',
    'created_on': '2024-01-01T00:00:00Z',
    'pycd_version': '1.0.0',
    'total_trajectories': 10,
    'format_version': 'hdf5_v1'
}

# System-level metadata  
system_info = {
    'lattice_matrix': np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]),
    'pbc': [1, 1, 1],
    'system_size': [2, 2, 1],
    'species_count': [5, 5],
    'total_species': 10,
    'temperature': 300.0
}

# Simulation parameters
simulation_params = {
    'time_interval': 1e-12,
    't_final': 1e-9,
    'n_traj': 10,
    'random_seed': 12345
}
```

## 3. Dual-Writing Strategy

### 3.1 Implementation Approach
1. **Configuration Control**: Add settings to enable/disable HDF5 writing
2. **Parallel Writing**: Write to both formats simultaneously during transition
3. **Data Validation**: Cross-check data integrity between formats
4. **Performance Monitoring**: Track overhead of dual-writing

### 3.2 Configuration Options
```python
# New configuration parameters
output_config = {
    'formats': {
        'npy': True,        # Continue writing .npy files
        'hdf5': False,      # Enable HDF5 writing
    },
    'hdf5_options': {
        'compression': 'gzip',
        'compression_opts': 6,
        'chunks': True,
        'shuffle': True,
        'fletcher32': False
    },
    'validation': {
        'cross_format_check': True,  # Validate npy vs hdf5 data
        'tolerance': 1e-15
    }
}
```

### 3.3 File Naming Convention
```
# Existing .npy files (unchanged)
traj1/unwrapped_traj.npy
traj1/wrapped_traj.npy
...

# New HDF5 files
trajectory_data.h5           # All trajectories in single file
# OR
traj1/trajectory.h5         # One HDF5 file per trajectory
traj2/trajectory.h5
...
```

## 4. API Changes and Code Modifications

### 4.1 Core Changes Required

#### 4.1.1 New HDF5 Writer Class
```python
class HDF5TrajectoryWriter:
    """Handle HDF5 trajectory writing operations"""
    
    def __init__(self, file_path, metadata, compression_opts):
        """Initialize HDF5 file with metadata"""
        
    def create_trajectory_group(self, traj_id, estimated_frames):
        """Create a new trajectory group with datasets"""
        
    def append_frame(self, traj_id, frame_data):
        """Append new trajectory frame data"""
        
    def finalize(self):
        """Close file and finalize datasets"""
```

#### 4.1.2 Modified Core Method
```python
# In Run.do_kmc_steps()
def write_trajectory_data(self, traj_dir_path, output_data, arrays):
    """Write trajectory data to configured formats"""
    
    # Existing .npy writing (if enabled)
    if self.config['formats']['npy']:
        self._write_npy_format(traj_dir_path, output_data, arrays)
    
    # New HDF5 writing (if enabled)
    if self.config['formats']['hdf5']:
        self._write_hdf5_format(traj_dir_path, output_data, arrays)
    
    # Cross-validation (if enabled)
    if self.config['validation']['cross_format_check']:
        self._validate_formats(traj_dir_path, arrays)
```

### 4.2 Configuration Integration
- Extend existing YAML configuration to include format options
- Add command-line arguments for format selection
- Environment variable support for CI/testing

### 4.3 Backward Compatibility
- All existing `.npy` reading code remains unchanged
- New HDF5 reading utilities provided as optional alternatives
- Gradual migration of analysis tools to support both formats

## 5. Testing and Validation Strategy

### 5.1 Data Integrity Tests
```python
def test_dual_format_equivalence():
    """Verify .npy and HDF5 contain identical data"""
    
def test_hdf5_metadata_accuracy():
    """Validate metadata matches simulation parameters"""
    
def test_incremental_writing():
    """Ensure append operations work correctly"""
    
def test_compression_integrity():
    """Verify compressed data matches uncompressed"""
```

### 5.2 Performance Tests
```python  
def test_write_performance():
    """Compare writing speed: npy vs hdf5 vs dual"""
    
def test_read_performance():
    """Compare reading speed for analysis operations"""
    
def test_storage_efficiency():
    """Compare file sizes with compression"""
```

### 5.3 Cross-Format Validation
- Automated comparison of trajectory data between formats
- Statistical validation (mean, variance, extrema)
- Binary-level comparison where appropriate
- Tolerance-based floating-point comparison

### 5.4 Regression Testing
- Ensure existing analysis workflows continue to work
- Validate unwrapping algorithms with both formats
- Test multi-species trajectory handling
- Verify PBC handling consistency

## 6. Migration Timeline and Phases

### 6.1 Phase 1: Foundation (Weeks 1-2)
- Implement HDF5 writer class
- Add configuration system for format selection
- Create basic HDF5 schema structure
- Initial unit tests for HDF5 writing

### 6.2 Phase 2: Dual-Writing Implementation (Weeks 3-4)
- Integrate HDF5 writing into core simulation loop
- Implement cross-format validation
- Comprehensive testing of dual-writing
- Performance benchmarking

### 6.3 Phase 3: Analysis Tool Updates (Weeks 5-6)
- Create HDF5 reading utilities
- Update analysis workflows to support HDF5
- Migration of existing scripts and examples
- Documentation updates

### 6.4 Phase 4: Validation and Optimization (Weeks 7-8)
- Large-scale testing with real simulations
- Performance optimization
- User feedback collection
- Bug fixes and refinements

### 6.5 Phase 5: Deprecation Planning (Weeks 9-12)
- Gradual shift to HDF5 as primary format
- Deprecation warnings for .npy-only workflows
- Migration assistance for users
- Final documentation and examples

## 7. User Communication and Documentation

### 7.1 User-Facing Changes
- **Configuration**: New YAML/CLI options for format selection
- **File Output**: Optional HDF5 files alongside existing .npy files
- **Analysis Tools**: New HDF5-compatible utilities
- **Performance**: Potentially improved storage efficiency

### 7.2 Documentation Requirements
- Migration guide for existing users
- HDF5 schema documentation for external tools
- Performance comparison and recommendations
- Troubleshooting guide for format-related issues

### 7.3 Communication Strategy
- Advance notice in release notes
- Blog post/tutorial on HDF5 benefits
- Example scripts showing format comparison
- Migration assistance for heavy users

## 8. Risk Assessment and Mitigation

### 8.1 Technical Risks
- **Performance Impact**: Dual-writing overhead
  - *Mitigation*: Benchmarking, optional enablement
- **Storage Requirements**: Temporary doubling of storage needs
  - *Mitigation*: Compression, user guidance on cleanup
- **Complexity**: Additional code paths and potential bugs
  - *Mitigation*: Comprehensive testing, gradual rollout

### 8.2 User Adoption Risks
- **Learning Curve**: New format and tools
  - *Mitigation*: Excellent documentation, backward compatibility
- **Workflow Disruption**: Changes to existing scripts
  - *Mitigation*: Long transition period, migration assistance
- **Tool Compatibility**: External tools expecting .npy format
  - *Mitigation*: Continue .npy support, provide conversion tools

## 9. Success Metrics

### 9.1 Technical Metrics
- Data integrity: 100% equivalence between formats
- Performance: <20% overhead for dual-writing
- Storage efficiency: >30% compression improvement
- Test coverage: >95% for new HDF5 functionality

### 9.2 User Metrics
- Migration rate: >50% users trying HDF5 within 6 months
- Issue reports: <5% increase due to format changes
- Documentation satisfaction: >90% helpful rating
- Community feedback: Positive reception for improved interoperability

## 10. Future Considerations

### 10.1 Advanced Features
- **Parallel I/O**: MPI-compatible HDF5 writing for large simulations
- **Streaming**: Real-time trajectory analysis during simulation
- **Compression**: Advanced compression algorithms (blosc, etc.)
- **Indexing**: Fast trajectory frame lookup and querying

### 10.2 Interoperability
- **Standard Formats**: Compatibility with MDAnalysis, VMD, etc.
- **Cloud Storage**: Efficient handling in cloud environments
- **Database Integration**: Trajectory metadata in relational databases
- **Version Control**: Handling schema evolution and backward compatibility

## Conclusion

This migration plan provides a structured approach to transitioning PyCD from NumPy to HDF5 trajectory storage while maintaining backward compatibility and ensuring data integrity. The dual-writing strategy minimizes risk while providing users time to adapt to the new format. The proposed HDF5 schema is designed for efficiency, interoperability, and future extensibility.

The success of this migration will depend on careful implementation, comprehensive testing, and clear communication with the user community. By following this plan, PyCD can achieve improved storage efficiency, better metadata handling, and enhanced interoperability with the broader scientific computing ecosystem.