# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
HDF5 trajectory I/O module for MDTraj-compliant trajectory storage.

This module provides functionality to save trajectory data in HDF5 format
following the MDTraj specification for maximum interoperability.
"""

import numpy as np
import h5py
from pathlib import Path

from PyCD import constants


class HDF5TrajectoryWriter:
    """
    Writer for MDTraj-compliant HDF5 trajectory format.
    
    The HDF5 format follows the MDTraj specification:
    - /coordinates: shape (n_frames, n_atoms, 3) in nanometers
    - /time: shape (n_frames,) in picoseconds
    - /topology/atoms: atom information
    """
    
    def __init__(self, filename, n_atoms, n_species, append=False):
        """
        Initialize HDF5 trajectory writer.
        
        Parameters
        ----------
        filename : str or Path
            Output HDF5 file path
        n_atoms : int
            Number of atoms/particles in the system
        n_species : int
            Number of species types
        append : bool, default False
            Whether to append to existing file or create new
        """
        self.filename = Path(filename)
        self.n_atoms = n_atoms
        self.n_species = n_species
        self.append = append
        self._frame_count = 0
        
        # Create or open HDF5 file
        mode = 'a' if append and self.filename.exists() else 'w'
        self.h5file = h5py.File(self.filename, mode)
        
        # Initialize datasets if creating new file
        if mode == 'w':
            self._create_topology()
            self._setup_datasets()
    
    def _create_topology(self):
        """Create topology group with atom information."""
        if 'topology' in self.h5file:
            del self.h5file['topology']
            
        topo_group = self.h5file.create_group('topology')
        
        # Create atoms table - simplified for now
        # In a full implementation, this would include more detailed atom info
        atoms_group = topo_group.create_group('atoms')
        
        # Create datasets for atom properties
        atom_names = np.array([f'ATOM{i}' for i in range(self.n_atoms)], dtype='S10')
        atom_elements = np.array(['C'] * self.n_atoms, dtype='S2')  # Default to carbon
        atom_indices = np.arange(self.n_atoms, dtype=np.int32)
        
        atoms_group.create_dataset('name', data=atom_names)
        atoms_group.create_dataset('element', data=atom_elements)
        atoms_group.create_dataset('index', data=atom_indices)
    
    def _setup_datasets(self):
        """Setup coordinate and time datasets with unlimited frames."""
        # Create coordinates dataset: (n_frames, n_atoms, 3) in nanometers
        self.coords_dset = self.h5file.create_dataset(
            'coordinates', 
            shape=(0, self.n_atoms, 3),
            maxshape=(None, self.n_atoms, 3),
            dtype=np.float32,
            chunks=True,
            compression='gzip'
        )
        self.coords_dset.attrs['units'] = 'nanometers'
        
        # Create time dataset: (n_frames,) in picoseconds
        self.time_dset = self.h5file.create_dataset(
            'time',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression='gzip'
        )
        self.time_dset.attrs['units'] = 'picoseconds'
    
    def write_frame(self, coordinates, time_value):
        """
        Write a single frame to the HDF5 file.
        
        Parameters
        ----------
        coordinates : array_like
            Coordinate data, shape (n_atoms, 3) or (n_atoms*3,) in Bohr
        time_value : float
            Time value in simulation units
        """
        # Convert coordinates to proper format
        coords = np.asarray(coordinates)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 3)
        
        # Convert from Bohr to nanometers (MDTraj requirement)
        coords_nm = coords / constants.ANG2BOHR * 0.1  # Bohr -> Angstrom -> nm
        
        # Convert time to picoseconds (MDTraj requirement)
        # Assuming input time is in atomic time units
        time_ps = time_value * constants.AUTIME2PS
        
        # Extend datasets
        self.coords_dset.resize((self._frame_count + 1, self.n_atoms, 3))
        self.time_dset.resize((self._frame_count + 1,))
        
        # Write data
        self.coords_dset[self._frame_count] = coords_nm
        self.time_dset[self._frame_count] = time_ps
        
        self._frame_count += 1
    
    def write_frames(self, coordinates_array, time_array):
        """
        Write multiple frames to the HDF5 file.
        
        Parameters
        ----------
        coordinates_array : array_like
            Coordinate data, shape (n_frames, n_atoms, 3) or (n_frames, n_atoms*3)
        time_array : array_like
            Time values, shape (n_frames,)
        """
        coords = np.asarray(coordinates_array)
        times = np.asarray(time_array)
        
        # Reshape coordinates if needed
        if coords.ndim == 2 and coords.shape[1] == self.n_atoms * 3:
            coords = coords.reshape(coords.shape[0], self.n_atoms, 3)
        
        n_frames = coords.shape[0]
        
        # Convert units
        coords_nm = coords / constants.ANG2BOHR * 0.1  # Bohr -> nm
        times_ps = times * constants.AUTIME2PS  # AU time -> ps
        
        # Extend datasets
        start_frame = self._frame_count
        end_frame = self._frame_count + n_frames
        
        self.coords_dset.resize((end_frame, self.n_atoms, 3))
        self.time_dset.resize((end_frame,))
        
        # Write data
        self.coords_dset[start_frame:end_frame] = coords_nm
        self.time_dset[start_frame:end_frame] = times_ps
        
        self._frame_count += n_frames
    
    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'h5file') and self.h5file:
            self.h5file.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def save_trajectory_hdf5(filename, coordinates, time_data=None, n_atoms=None):
    """
    Convenience function to save trajectory data to HDF5 format.
    
    Parameters
    ----------
    filename : str or Path
        Output HDF5 file path
    coordinates : array_like
        Coordinate data in Bohr units
    time_data : array_like, optional
        Time data in simulation units
    n_atoms : int, optional
        Number of atoms, inferred from coordinates if not provided
    """
    coords = np.asarray(coordinates)
    
    # Infer n_atoms if not provided
    if n_atoms is None:
        if coords.ndim == 1:
            n_atoms = len(coords) // 3
        elif coords.ndim == 2:
            n_atoms = coords.shape[1] // 3 if coords.shape[1] % 3 == 0 else coords.shape[1]
        else:
            raise ValueError("Cannot infer n_atoms from coordinate shape")
    
    # Create dummy time data if not provided
    if time_data is None:
        if coords.ndim == 1:
            time_data = np.array([0.0])
        else:
            time_data = np.arange(coords.shape[0], dtype=float)
    
    with HDF5TrajectoryWriter(filename, n_atoms, 1) as writer:
        if coords.ndim == 1:
            writer.write_frame(coords, time_data[0])
        else:
            writer.write_frames(coords, time_data)