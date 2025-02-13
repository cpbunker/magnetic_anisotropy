import os
from os import path
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

eV2wavenumber = 8065.61;
eV2meV = 1e3;
plt.rcParams.update({"font.family":"serif"});
plt.rcParams.update({"font.family":"Times New Roman"});
plt.rcParams.update({"font.size":12});

def polar_plot(base_name = "energies", title="", columns=(0, 1, 6), factor=1.0, wavenumber=True):
    '''
    Energies from input file are in eV
    Plot them in either cm^-1 (case wavenumber=True) or meV (case wavenumber=False)
    factor: scale factor for the energy. useful when multiple equivalent spins are rotated together.
    '''

    columns = np.array(columns, dtype=int) - 1

    ### Input and output
    
    fin = base_name + ".dat"
    fout = base_name + ".pdf"

    if not os.path.isfile(fin):
        raise Exception(fin+" does not exist");
    
    energies = np.loadtxt(fin) 
    print(np.shape(energies));
    
    
    ### Data massage

    column1 = columns[0]
    column2 = columns[1]
    column3 = columns[2]
    
    emin = np.nanmin(energies[:, column3])
    
    nconfig = energies.shape[0]
    
    for i in range(nconfig):
        if energies[i, column3] == 0.0:
            energies[i, column3] = np.nan

        # handle units. Recall energies are input in eV
        else:
            if wavenumber:
                energies[i, column3] = factor* eV2wavenumber*(energies[i, column3] - emin)
            else:
                energies[i, column3] = factor* eV2meV*(energies[i, column3] - emin)
    
    emin = np.nanmin(energies[:, column3])
    emax = np.nanmax(energies[:, column3])

    # convert theta (column 1) and phi (column 2) to degrees
    energies[:, column1] = np.deg2rad(energies[:, column1])
    energies[:, column2] = np.deg2rad(energies[:, column2])

    
    
    ### Plot
    
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = fig.add_subplot(projection="polar")
    
    ax.set_title(title, {'fontsize': 13})
    
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([0, np.pi])

    ax.set_xticks(np.linspace(0, 2*np.pi, 9, endpoint=True), labels=[r"$\phi=0$", r"", r"$\phi=\pi/2$", r"", r"$\phi=\pi$", r"", r"$\phi=3\pi/2$", r"", r""])
    ax.set_yticks(np.linspace(0,   np.pi, 7, endpoint=True), labels=[r"$\theta=0$", r"", r"", r"$\theta=\pi/2$", r"", r"", r"$\theta=\pi$"])

    # scatter phi, theta, energy points (use c=energy)
    cm = mpl.colormaps['rainbow']
    scat = ax.scatter(energies[:,column2], energies[:,column1], s=60, c=energies[:,column3], vmin=emin, vmax=emax, cmap=cm)
    
    if(wavenumber):
        cbar = plt.colorbar(scat, label = r"$E(\theta, \phi)\; (\mathrm{cm}^{-1})$", pad=0.15)
    else:
        cbar = plt.colorbar(scat, label = r"$E(\theta, \phi)\; (\mathrm{meV})$", pad=0.15)

    fig.tight_layout()
    
    plt.savefig(fout)

if __name__ == "__main__":

    # change to working directory
    existing_dat = sys.argv[1];
    if(path.exists(existing_dat)): pass
    else: raise Exception(existing_dat+" must already exist!");
    #print(">>> pwd");
    #subprocess.run(["pwd"]);
    #print(">>> ls "+root_dir);
    #subprocess.run(["ls", root_dir]);

    # pdf of polar energies
    polar_plot(base_name = existing_dat[:-4], title="", columns=(1,2,3), factor=1, wavenumber=False)

