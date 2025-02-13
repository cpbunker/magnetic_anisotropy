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
Ang2met = 1e-10;
met2Ang = 1e10;
Hz2microeV = 4.136e-9
Hz2GHz = 1e-9;
meV2microeV = 1e3;

# Spin operators for S = 1/2
S = [0.5*np.array([[0,1],[1,0]]), 0.5*np.array([[0,-1j],[1j,0]]), 0.5*np.array([[1,0],[0,-1]])]

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

def get_A(r1, r2, print_in_GHz = False):
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

    if(print_in_GHz):
        print("The magnetic dipolar coupling matrix in GHz is \n"); print(A*Hz2GHz); print()
    else:
        print("The magnetic dipolar coupling matrix in microeV is \n"); print(A*Hz2microeV); print()
    return A # <- in Hz

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

    return H # <- in Hz

def get_diff_E_xyz(r1, r2):
    """
    Get the energy difference between the eigenstates of Sx, Sy, and Sz.
    """
    raise NotImplementedError

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

def get_diff_E_en(r1, r2, en1, en2, verbose=True):
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
    dE_GHz = dE_Hz*Hz2GHz
    dE_J =  dE_Hz * h
    dE_microeV = dE_J * Joule2meV*meV2microeV
    dE_wavenumber = dE_J * Joule2wavenumber

    if(verbose):
        print("The energy difference E_ud - E_uu is:")
        print("{:12.6f} GHz".format(dE_GHz))
        print("{:12.6f} microeV".format(dE_microeV))
        print("{:12.6f} cm^-1".format(dE_wavenumber))
        print()

def get_E_classical(r1, r2, en1, en2, verbose=True):
    '''
    r1 and r2 are position vectors, in meters
    '''
    for en in [en1, en2]:
        if(abs(1-np.linalg.norm(en)) > 1e-10): raise ValueError;

    # r vector
    rnorm = np.linalg.norm(r2-r1); # <- in meters
    rA = rnorm*met2Ang;            # <- in angstroms
    rhat = (r2-r1)/rnorm;

    # potential energy
    classical_prefactor = -(53.69/Hz2microeV)/(rA**3); # <- in Hz
    dE_Hz = classical_prefactor*(3*np.dot(rhat, en1)*np.dot(rhat, en2) - np.dot(en1, en2)); # <- still in Hz
    if(verbose):
        print("The classical magnetic dipole interaction energy is:")
        print("{:12.6f} GHz".format(dE_Hz*Hz2GHz))
        print("{:12.6f} microeV".format(dE_Hz*Hz2microeV))
        print("{:12.6f} cm^-1".format(dE_Hz * h * Joule2wavenumber))
        print()
    return dE_Hz;

def circular_E_classical(r1, r2, en1_init, en2, plane_definer, num_alpha=12, wavenumber = False):
    '''
    '''
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif"});
    plt.rcParams.update({"font.family":"Times New Roman"});
    plt.rcParams.update({"font.size":12});

    # set up coordinate system for the molecular plane
    exprime = en1_init/np.linalg.norm(en1_init);
    eyprime = plane_definer - np.dot(plane_definer,exprime)*exprime;
    eyprime = eyprime/np.linalg.norm(eyprime);
    print("exprime dot eyprime = {:.6f}".format(np.dot(exprime, eyprime)));

    # rotate en1 around the molecular plane
    E_class_vals = np.zeros((num_alpha),dtype=float);
    alpha_vals = np.linspace(0, 2*np.pi, num_alpha+1)[:-1];
    theta_fixed = np.pi/2*np.ones_like(alpha_vals); # keeps us in the xy plane of the polar projection,
                                                    # which is the exprime-eyprime plane

    for vali in range(num_alpha):
        en1_alpha = exprime*np.cos(alpha_vals[vali]) + eyprime*np.sin(alpha_vals[vali]);
        E_class_vals[vali] = get_E_classical(r1, r2, en1_alpha, en2, verbose=False); # <- in Hz
        #print("E(rA={:3.2f}".format(np.linalg.norm(r2-r1)));
        #print("alpha={:.2f}*pi) ".format( alpha_vals/np.pi));
        #print("{:.6f} micro eV".format( Hz2microeV*E_class_vals[vali]));
        
        print("E(rA={:3.2f}, alpha={:.2f}*pi) = {:.6f} micro eV".format(np.linalg.norm(r2-r1)*met2Ang,
                                                                alpha_vals[vali]/np.pi, Hz2microeV*E_class_vals[vali]));

    # convert E_class_vals units
    if(wavenumber): E_class_vals = E_class_vals * h * Joule2wavenumber; # < in cm^-1
    else:           E_class_vals = E_class_vals * Hz2microeV;           # <- in micro eV
 
    # polar plot
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = fig.add_subplot(projection="polar")
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([0, np.pi])
    ax.set_xticks(np.linspace(0, 2*np.pi, 9, endpoint=True), labels=[r"$\phi=0$", r"", r"$\phi=\pi/2$", r"", r"$\phi=\pi$", r"", r"$\phi=3\pi/2$", r"", r""])
    ax.set_yticks(np.linspace(0,   np.pi, 7, endpoint=True), labels=[r"$\theta=0$", r"", r"", r"$\theta=\pi/2$", r"", r"", r"$\theta=\pi$"])

    # scatter phi, theta, energy points (use c=energy)
    cm = plt.colormaps['rainbow']
    scat = ax.scatter(alpha_vals, theta_fixed, c=E_class_vals,  
                      s=60, vmin=np.nanmin(E_class_vals), vmax=np.nanmax(E_class_vals), cmap=cm)
    
    if(wavenumber):
        cbar = plt.colorbar(scat, label = r"$E(\theta, \phi)\; (\mathrm{cm}^{-1})$", pad=0.15)
    else:
        cbar = plt.colorbar(scat, label = r"$E(\theta, \phi)\; (\mathrm{\mu eV})$", pad=0.15)

    fig.tight_layout()
    plt.show();

def do_dimers(vaspfiles, whichselect):
    '''
    For DHP,    use whichselect=4 to select N5, the correct N relative to Cu2
    For tbutyl, use whichselect=7 to select N8 
    '''
    for i in range(len(vaspfiles)):
        print("\nExamining the system {:s}\n".format(vaspfiles[i]))
        atoms = read(vaspfiles[i])
        atoms = set_z_axis(atoms)
        Cu1 = atoms[0].position # <- in Ang
        Cu2 = atoms[1].position # <- in Ang

        # grab the nitrogens N1 and N5 for bond directions
        whichatomstart = 2; # first two atoms are Cu1 and Cu2
        # For DHP, I verified that cycling through the 4 N atoms ringing each Cu
        # does not change E_ud-E_uu by more than 0.001 micro eV !!
        N1 = atoms[whichatomstart].position;
        Nselected = atoms[whichatomstart+whichselect].position;

        # ASSUMING a molecular easy plane, these bond directions -> quantization axes
        en1 = N1 - Cu1; 
        en1 = en1 / np.linalg.norm(en1)
        en2 = Nselected - Cu2; 
        en2 = en2 / np.linalg.norm(en2)
        # uses another Cu-N bond of 1st Cu to define the molecular plane
        defines_inplane = (atoms[whichatomstart+1].position - Cu1)/np.linalg.norm(atoms[whichatomstart+1].position - Cu1);
        outofplane = np.cross( en1, defines_inplane); # uses 
        outofplane = outofplane / np.linalg.norm(outofplane);
        print("en1 dot en2 = {:.2f}".format(np.dot(en1,en2)));
        print("en1 outofplane = {:.2f}".format(np.dot(en1,outofplane)));
        print("en2 outofplane = {:.2f}".format(np.dot(en2,outofplane)));
        if( abs(np.dot(en1,en2)) < 0.5):
            print("en1 dot en2 = {:.2f}".format(np.dot(en1,en2)));
            raise Exception("change argument whichatom to better align quantization axes")
        
        # tell the user the locations and the quantum energy
        print("The location of metal 1 is r1 = ", Cu1, "Ang")
        print("The location of metal 2 is r2 = ", Cu2, "Ang")
        print("The metal-metal separation is {:.2f} Ang".format(np.linalg.norm(Cu1-Cu2)))
        print("The quantization axis for S1 (for vanadium1) is = ", en1)
        print("The quantization axis for S2 (for vanadium2) is = ", en2)
        print()
        get_diff_E_en(Cu1*Ang2met, Cu2*Ang2met, en1, en2)
        print()

        # tell the user the classical energy
        get_E_classical(Cu1*Ang2met, Cu2*Ang2met, en1, en2);

        # plot the classical energy as the Cu1 spin rotates in its molecular plane
        circular_E_classical(Cu1*Ang2met, Cu2*Ang2met, en1, en2, defines_inplane)


if __name__ == "__main__":
    # Reference result for r12 = 3 Ang: 1.9 GHz, 1.9 GHz, and -3.8 GHz in the diagonal matrix elements
    #                              i.e. 7.9mueV, 7.9mueV, and -15.7 mueV
    if(True):
        ref_amt = (27.5074-10.5561) # <- in ang, for DHP closed
        ref_amt = (28.2996-11.6003) # <- in ang, for DHP open
        print("\nExamining reference separation of {:.2f} Ang\n".format(ref_amt))
        r1_ref = np.array([0, 0, 0]);
        r2_ref = np.array([0, 0, ref_amt*Ang2met]);

        # choose the quantization axes
        quant1_ref = np.array([0,0,1]);
        # E_ud - E_uu is very sensitive to this choice, see e.g.
        #quant1_ref = np.array([0,np.sqrt(1/2),np.sqrt(1/2)]);
        quant2_ref = 1*quant1_ref; # try (-1)
        print("quant1_ref dot qant2_ref = {:.2f}".format(np.dot(quant1_ref, quant2_ref)));

        # run reference energy - quantum
        get_diff_E_en(r1_ref, r2_ref, quant1_ref, quant2_ref);

        # run reference energy - classical
        get_E_classical(r1_ref, r2_ref, quant1_ref, quant2_ref)
        del ref_amt, r1_ref, r2_ref, quant1_ref, quant2_ref;

    do_dimers(sys.argv[2:], int(sys.argv[1]));
