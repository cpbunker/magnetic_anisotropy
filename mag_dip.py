# ===========================================================
# 1. Get the magnetic dipolar coupling matrix
# 2. Get the energy difference between two given spin states.
# ===========================================================

import numpy as np
from ase.io import read

import sys

e = 1.602176634e-19 # Coulombs
m_e = 9.1093837139e-31 # Kg
g_e = -2 # g factor for electrons

gamma_e = -e/(2*m_e) * abs(g_e) # gyromagnetic ratio for electrons in rad s^-1 T^-1
gamma_e_freq = gamma_e / (2*np.pi) # gyromagnetic ratio for electrons in Hz T^-1

mu_0 = 1.25663706127e-6 # N * A^-2

h = 6.62607015e-34 # J Hz^-1
hbar = h / (2*np.pi)

Joule2eV = 6.241509074460763e+18
Joule2meV = 6.241509074460763e+21
meV2wavenumber = 8.06554393738
Joule2wavenumber = 5.034116567561925e+22 # Joule2meV * meV2wavenumber
m2Ang = 1e-10;
Ang2m = 1e10;

# Spin operators for S = 1/2
S = [0.5*np.array([[0,1],[1,0]]), 0.5*np.array([[0,-1j],[1j,0]]), 0.5*np.array([[1,0],[0,-1]])]

def get_A(r1, r2):
    """
    A: Magnetic dipolar coupling matrix

    H = h S_1 \cdot A \cdot S_2 :: unit J (most clear, h is the Planck constant)
    H =   S_1 \cdot A \cdot S_2 :: unit Hz (practical, to be used here)

    Units:
      A and H: Hz
      r1 and r2: m
    """

    r1 = np.array(r1)
    r2 = np.array(r2)

    gamma_1 = gamma_e_freq
    gamma_2 = gamma_e_freq

    r12 = r2 - r1
    r12_norm = np.linalg.norm(r12)

    rcircr = np.reshape(np.kron(r12, r12), (3,3))

    A = - gamma_1 * gamma_2 * (mu_0 * hbar) / (2*r12_norm**5) * (3*rcircr - r12_norm**2 * np.eye(3))

    print("The magnetic dipolar coupling matrix in Hz is \n"); print(A); print()

    return A

def get_H(r1, r2):
    """
    Get H =   S_1 \cdot A \cdot S_2
    """

    A = get_A(r1, r2)

    S1 = S
    S2 = S

    H = np.zeros((4,4))
    for i in range(3):
        for j in range(3):
            H = H + A[i, j]*np.kron(S1[i], S2[j])

    #print("The Hamiltonian is"); print(H)

    return H

def get_diff_E_xyz(r1, r2):
    """
    Get the energy difference between the eigenstates of Sx, Sy, and Sz.
    """

    eigenvalues_x, eigenvectors_x = np.linalg.eigh(S[0])
    eigenvalues_y, eigenvectors_y = np.linalg.eigh(S[1])
    eigenvalues_z, eigenvectors_z = np.linalg.eigh(S[2])

    #print(eigenvalues_x)
    #print(eigenvalues_y)
    #print(eigenvalues_z)

    eigenvectors_fm = [ \
            np.kron(eigenvectors_x[:, 1], eigenvectors_x[:, 1]), \
            np.kron(eigenvectors_y[:, 1], eigenvectors_y[:, 1]), \
            np.kron(eigenvectors_z[:, 1], eigenvectors_z[:, 1]), \
    ]

    eigenvectors_af = [ \
            np.kron(eigenvectors_x[:, 1], eigenvectors_x[:, 0]), \
            np.kron(eigenvectors_y[:, 1], eigenvectors_y[:, 0]), \
            np.kron(eigenvectors_z[:, 1], eigenvectors_z[:, 0]), \
    ]

    tags = ["x", "y", "z"]

    H = get_H(r1, r2)

    for i in range(3):
        E_fm = np.dot( np.conjugate(eigenvectors_fm[i]), np.matmul(H, eigenvectors_fm[i]) )
        E_af = np.dot( np.conjugate(eigenvectors_af[i]), np.matmul(H, eigenvectors_af[i]) )
        dE_Hz = np.real( E_af - E_fm )
        dE_GHz = dE_Hz/10**9
        dE_J =  dE_Hz * h
        dE_meV = dE_J * Joule2meV
        dE_wavenumber = dE_J * Joule2wavenumber

        print("When the spins are along the {:s} directions, the energy difference E_AF - E_FM is".format(tags[i]))
        print("{:12.6f} GHz".format(dE_GHz))
        print("{:12.6f} meV".format(dE_meV))
        print("{:12.6f} cm^-1".format(dE_wavenumber))
        print()

def get_diff_E_en(r1, r2, en1, en2):
    """
    Get the energy difference between |en1, en2> and |en1, -en2>.
    """

    Sn1 = en1[0]*S[0] + en1[1]*S[1] + en1[2]*S[2]
    Sn2 = en2[0]*S[0] + en2[1]*S[1] + en2[2]*S[2]

    eigenvalues_1, eigenvectors_1 = np.linalg.eigh(Sn1)
    eigenvalues_2, eigenvectors_2 = np.linalg.eigh(Sn2)

    #print(eigenvalues_1)
    #print(eigenvalues_2)

    eigenvector_uu = np.kron(eigenvectors_1[:, 1], eigenvectors_2[:, 1])
    eigenvector_ud = np.kron(eigenvectors_1[:, 1], eigenvectors_2[:, 0])

    H = get_H(r1, r2)

    E_uu = np.dot( np.conjugate(eigenvector_uu), np.matmul(H, eigenvector_uu) )
    E_ud = np.dot( np.conjugate(eigenvector_ud), np.matmul(H, eigenvector_ud) )
    dE_Hz = np.real( E_ud - E_uu )
    dE_GHz = dE_Hz/10**9
    dE_J =  dE_Hz * h
    dE_meV = dE_J * Joule2meV
    dE_wavenumber = dE_J * Joule2wavenumber

    print("The energy difference E_ud - E_uu is")
    print("{:12.6f} GHz".format(dE_GHz))
    print("{:12.5f} meV".format(dE_meV))
    print("{:12.4f} cm^-1".format(dE_wavenumber))
    print()

def set_z_axis(atoms):
    '''
    Sets the z axis of the LAB frame along the Cu-Cu direction
    '''
    p1 = atoms[0].position
    p2 = atoms[1].position

    ez = p2 - p1
    ez = ez / np.linalg.norm(ez)

    x = np.dot(ez, [1, 0, 0])
    if abs(x) < 0.999:
        ex = np.cross([1, 0, 0], ez)
    else:
        ex = np.cross([0, 1, 0], ez)
    ex = ex / np.linalg.norm(ex)

    ey = np.cross(ez, ex)

    emat = [ex, ey, ez]

    positions = atoms.get_positions()
    positions = np.transpose(positions)
    positions = emat @ positions
    positions = np.transpose(positions)

    atoms.set_positions(positions)

    return atoms

def do_dimers(vaspfiles, whichatom=3):
    for i in range(len(vaspfiles)):
        print("Examining the system {:s}\n".format(vaspfiles[i]))
        atoms = read(vaspfiles[i])
        atoms = set_z_axis(atoms)
        Cu1 = atoms[0].position
        Cu2 = atoms[1].position
        # quantization axes ASSUMING a molecular easy plane
        en1 = atoms[2].position - Cu1; 
        en1 = en1 / np.linalg.norm(en1)
        en2 = atoms[whichatom].position - Cu2; 
        en2 = en2 / np.linalg.norm(en2)
        print("en1 dot en2 = {:.2f}".format(np.dot(en1,en2)));
        if( abs(np.dot(en1,en2)) < 0.5):
            print("en1 dot en2 = {:.2f}".format(np.dot(en1,en2)));
            raise Exception("change argument whichatom to better align quantization axes")
        print("The location of metal 1 is r1 = ", Cu1, "Ang")
        print("The location of metal 2 is r2 = ", Cu2, "Ang")
        print("The metal-metal separation is {:.2f} Ang".format(np.linalg.norm(Cu1-Cu2)))
        print("The quantization axis for S1 (for vanadium1) is = ", en1)
        print("The quantization axis for S2 (for vanadium2) is = ", en2)
        print()
        get_diff_E_en(Cu1*m2Ang, Cu2*m2Ang, en1, en2)
        print("\n")

if __name__ == "__main__":
    # Reference result for r12 = 3 Ang: 1.9 GHz, 1.9 GHz, and -3.8 GHz in the diagonal matrix elements.
    if(True):
        ref_amt = (27.51-10.57)
        print("Examining reference separation of {:.2f} Ang\n".format(ref_amt))
        r1_ref = [0, 0, 0]
        r2_ref = [0, 0, ref_amt*m2Ang]
        quant1_ref = [0,0,1];
        get_diff_E_en(r1_ref, r2_ref, quant1_ref, quant1_ref)

    do_dimers(sys.argv[1:]);
