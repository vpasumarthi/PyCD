# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# CONSTANTS
EPSILON0 = 8.854187817E-12  # Electric constant in F.m-1
ANG = 1E-10  # Angstrom in m
KB = 1.38064852E-23  # Boltzmann constant in J/K
pi = 3.14159265358979323846  # Approximately

# FUNDAMENTAL ATOMIC UNITS
# Source: http://physics.nist.gov/cuu/Constants/Table/allascii.txt
EMASS = 9.10938356E-31  # Electron mass in Kg
ECHARGE = 1.6021766208E-19  # Elementary charge in C
HBAR = 1.054571800E-34  # Reduced Planck's constant in J.sec

# DERIVED ATOMIC UNITS
KE = 1 / (4 * pi * EPSILON0)
# Bohr radius in m
BOHR = HBAR ** 2 / (EMASS * ECHARGE ** 2 * KE)
# Hartree in J
HARTREE = HBAR ** 2 / (EMASS * BOHR ** 2)
AUTIME = HBAR / HARTREE  # sec
AUTEMPERATURE = HARTREE / KB  # K
AUPOT = HARTREE / ECHARGE  # V

# CONVERSIONS
EV2J = ECHARGE
ANG2BOHR = ANG / BOHR
ANG2UM = 1.00E-04
ANG2CM = 1.00e-08
J2HARTREE = 1 / HARTREE
SEC2AUTIME = 1 / AUTIME
V2AUPOT = 1 / AUPOT
EV2HARTREE = EV2J * J2HARTREE
SEC2NS = 1.00E+09
SEC2PS = 1.00E+12
SEC2FS = 1.00E+15
K2AUTEMP = 1 / AUTEMPERATURE
AUTIME2NS = SEC2NS / SEC2AUTIME
AUTIME2PS = SEC2PS / SEC2AUTIME
AUTIME2FS = SEC2FS / SEC2AUTIME
BOHR2UM = ANG2UM / ANG2BOHR
BOHR2CM = ANG2CM / ANG2BOHR