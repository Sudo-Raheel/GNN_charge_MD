from pymatgen.core import Molecule, Lattice, Structure
from pymatgen.io.xyz import XYZ


# Example usage:
A = 15.892 
B = 15.892 
C = 25.914
xyz_file = "struc.xyz"

# Load the molecule from the XYZ file
molecule = Molecule.from_file(xyz_file)
molecule=molecule.get_centered_molecule()
structure=molecule.get_boxed_structure(a=A,b=B,c=C)
# lattice = Lattice.from_parameters(A, B, C, 90, 90, 90)
# species=molecule.atomic_numbers
# coords= molecule.cart_coords
# structure = Structure(lattice, species, coords)
structure.to(fmt="poscar", filename='First.vasp')
# Print the molecule object or its structure

