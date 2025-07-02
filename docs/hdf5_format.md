# HDF5 Trajectory Format Support

PyCD now supports HDF5 trajectory output in addition to the traditional NumPy `.npy` format. The HDF5 format follows the MDTraj specification for maximum interoperability with molecular dynamics analysis tools.

## Configuration

HDF5 output is configured through the `output_data` section in your simulation parameters YAML file:

```yaml
output_data:
  # Traditional outputs (always present)
  unwrapped_traj: {file_name: unwrapped_traj.npy, write: 1, write_every_step: 0}
  time: {file_name: time_data.npy, write: 1}
  
  # NEW: HDF5 output (optional, enabled by default)
  hdf5_output: {enabled: true, file_name: trajectory.h5}
```

### Configuration Options

- `enabled` (boolean): Whether to generate HDF5 output. Default: `true`
- `file_name` (string): Name of the HDF5 trajectory file. Default: `trajectory.h5`

## Backward Compatibility

- **NumPy `.npy` format remains the primary output** - it cannot be disabled
- **HDF5 output is optional** and enabled by default for new simulations  
- **Existing configuration files** without HDF5 settings continue to work unchanged
- **No command-line options** - configuration is file-based only

## HDF5 Format Specification

The HDF5 trajectory files follow the MDTraj specification:

```
trajectory.h5
├── coordinates      # Dataset: (n_frames, n_atoms, 3) in nanometers
├── time             # Dataset: (n_frames,) in picoseconds  
└── topology/        # Group: Atom and topology information
    └── atoms/       # Group: Atom properties
        ├── name     # Dataset: Atom names
        ├── element  # Dataset: Element symbols
        └── index    # Dataset: Atom indices
```

### Units

- **Coordinates**: Nanometers (converted from Bohr automatically)
- **Time**: Picoseconds (converted from atomic time units automatically)

## Usage Examples

### Basic Configuration

Enable HDF5 output with default settings:

```yaml
output_data:
  unwrapped_traj: {file_name: unwrapped_traj.npy, write: 1}
  hdf5_output: {enabled: true, file_name: trajectory.h5}
```

### Disable HDF5 Output

For simulations where you only want NumPy output:

```yaml
output_data:
  unwrapped_traj: {file_name: unwrapped_traj.npy, write: 1}
  hdf5_output: {enabled: false, file_name: trajectory.h5}
```

### Reading HDF5 Trajectories

The HDF5 trajectories can be read with any MDTraj-compatible software:

```python
import h5py
import mdtraj as md

# Read with h5py
with h5py.File('trajectory.h5', 'r') as f:
    coordinates = f['coordinates'][:]  # (n_frames, n_atoms, 3) in nm
    time = f['time'][:]               # (n_frames,) in ps

# Or read with MDTraj (if topology is available)
traj = md.load('trajectory.h5')
```

## File Locations

- **Serial mode**: HDF5 files are saved in individual trajectory directories (`traj1/`, `traj2/`, etc.)
- **Parallel mode**: HDF5 files are saved in the main output directory

## Data Consistency

When HDF5 output is enabled, both formats contain equivalent trajectory data:

- **NumPy format**: Raw simulation coordinates in Bohr, time in atomic units
- **HDF5 format**: Coordinates converted to nanometers, time converted to picoseconds
- **Validation**: Cross-format consistency tests ensure data integrity

## Migration Considerations

- **Current stage**: Both formats are written simultaneously (when HDF5 enabled)
- **Future**: NumPy format may be deprecated in favor of HDF5
- **Recommendation**: Enable HDF5 output for new simulations to future-proof your workflows

## Error Handling

If HDF5 writing fails (e.g., due to disk space or permissions), the simulation continues and:

- A warning message is printed
- NumPy output continues normally
- The simulation is not interrupted

## Dependencies

HDF5 support requires the `h5py` package, which is automatically installed with PyCD:

```bash
pip install PyCD  # Includes h5py dependency
```

If `h5py` is not available, HDF5 output is silently disabled with a warning.