#!/usr/bin/env python3
"""
Command-line utility to unwrap wrapped trajectory files.

This script provides a simple interface to the unwrap_trajectory functionality
without requiring knowledge of the PyCD API.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

try:
    from PyCD.core import unwrap_trajectory_file
    from PyCD import constants
    from PyCD.io import read_poscar
except ImportError as e:
    print(f"Error: Failed to import PyCD modules: {e}")
    print("Make sure PyCD is properly installed.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Unwrap a wrapped trajectory file to obtain continuous coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with manual parameters
  python unwrap_trajectory.py wrapped_traj.npy --system-size 2 2 1 \\
    --lattice-diagonal 10.0 10.0 12.0 -o unwrapped_traj.npy

  # Using POSCAR file for lattice parameters
  python unwrap_trajectory.py wrapped_traj.npy --system-size 2 2 1 \\
    --poscar POSCAR -o unwrapped_traj.npy
    
  # Disable PBC in z-direction
  python unwrap_trajectory.py wrapped_traj.npy --system-size 2 2 1 \\
    --lattice-diagonal 10.0 10.0 12.0 --pbc 1 1 0 -o unwrapped_traj.npy
        """
    )
    
    parser.add_argument("input_file", type=str, 
                       help="Path to wrapped trajectory file (.npy format)")
    
    parser.add_argument("--system-size", type=float, nargs=3, required=True,
                       metavar=("NX", "NY", "NZ"),
                       help="System size in number of unit cells [nx, ny, nz]")
    
    # Lattice parameters - either diagonal or POSCAR
    lattice_group = parser.add_mutually_exclusive_group(required=True)
    lattice_group.add_argument("--lattice-diagonal", type=float, nargs=3,
                              metavar=("LX", "LY", "LZ"),
                              help="Lattice diagonal elements in Angstrom (for cubic/orthorhombic)")
    lattice_group.add_argument("--poscar", type=str,
                              help="Path to POSCAR file to read lattice matrix")
    
    parser.add_argument("--pbc", type=int, nargs=3, default=[1, 1, 1],
                       metavar=("X", "Y", "Z"),
                       help="Periodic boundary conditions [x, y, z] (0 or 1, default: 1 1 1)")
    
    parser.add_argument("-o", "--output", type=str,
                       help="Output file path for unwrapped trajectory")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    if args.verbose:
        print(f"Input file: {input_file}")
        print(f"System size: {args.system_size}")
        print(f"PBC: {args.pbc}")
    
    # Set up lattice matrix
    if args.lattice_diagonal:
        # Simple diagonal lattice matrix
        lattice_matrix = np.diag(args.lattice_diagonal) * constants.ANG2BOHR
        if args.verbose:
            print(f"Lattice diagonal (Angstrom): {args.lattice_diagonal}")
    else:
        # Read from POSCAR
        poscar_path = Path(args.poscar)
        if not poscar_path.exists():
            print(f"Error: POSCAR file '{poscar_path}' not found")
            sys.exit(1)
        
        poscar_info = read_poscar(poscar_path)
        lattice_matrix = poscar_info['lattice_matrix'] * constants.ANG2BOHR
        if args.verbose:
            print(f"Lattice matrix read from: {poscar_path}")
            print(f"Lattice matrix (Angstrom):")
            print(poscar_info['lattice_matrix'])
    
    # Unwrap trajectory
    try:
        if args.verbose:
            print("Unwrapping trajectory...")
        
        unwrapped_traj = unwrap_trajectory_file(
            input_file, 
            args.system_size, 
            lattice_matrix, 
            pbc=args.pbc,
            output_file_path=args.output
        )
        
        if args.verbose:
            print(f"Unwrapped trajectory shape: {unwrapped_traj.shape}")
            print(f"Trajectory range (Angstrom):")
            traj_ang = unwrapped_traj / constants.ANG2BOHR if unwrapped_traj.ndim == 2 else unwrapped_traj.reshape(-1, 3) / constants.ANG2BOHR
            for dim, axis in enumerate(['X', 'Y', 'Z']):
                if unwrapped_traj.ndim == 2:
                    coord_data = unwrapped_traj[:, dim] / constants.ANG2BOHR
                else:
                    coord_data = unwrapped_traj.reshape(-1, 3)[:, dim] / constants.ANG2BOHR
                print(f"  {axis}: {np.min(coord_data):.3f} to {np.max(coord_data):.3f}")
        
        if args.output:
            print(f"✓ Unwrapped trajectory saved to: {args.output}")
        else:
            print("✓ Unwrapping completed successfully")
            print(f"Return unwrapped trajectory array with shape {unwrapped_traj.shape}")
        
    except Exception as e:
        print(f"Error: Failed to unwrap trajectory: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()