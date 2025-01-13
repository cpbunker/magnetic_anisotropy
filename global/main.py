import os
import numpy as np
from os import path
import sys
import subprocess

# root dir
root_dir = sys.argv[1];
assert(root_dir[-1]=="/");
existing_poscar = root_dir+"POSCAR"
if(path.exists(existing_poscar)): pass
else: raise Exception(existing_poscar+" must already exist!");
existing_pot = root_dir+"POTCAR"
if(path.exists(existing_pot)): pass
else: raise Exception(existing_pot+" must already exist!");
existing_data = root_dir+"data.py"
if(path.exists(existing_data)): pass;
else: raise Exception(existing_data+" must already exist");
print(">>> pwd");
subprocess.run(["pwd"]);
print(">>> ls "+root_dir);
subprocess.run(["ls", root_dir]);

# data.py input file from root dir
subprocess.run(["cp", root_dir+"data.py","."]);
subprocess.run(["ls"]);

# wave dir
wave_dir = sys.argv[2];
assert("/" not in wave_dir);

from common import *

def find_global_directions():
    direction_emin = (71.4,  59.5)
    myjob = vasp_jobs_ncl(n_theta=1801, n_phi=3600)
    myjob.setup_local_ref_frames(z_direction=direction_emin, from_file=False)
    myjob.add_directions([[0.92, 180]], local_ref_frame=True)
    myjob.print_dirs_and_configs(local_ref_frame=False)
    #print("Opposite direction: {:6.1f} {:6.1f}".format(*opposite_direction))

def do(to_do, e_ref, de0, max_energy = np.inf):

    myjob = vasp_jobs_ncl(n_theta=1801, n_phi=3600)
    #myjob.setup_local_ref_frames(z_direction=direction_emin, from_file=True)


    #myjob.add_thetas(phi=180, theta_min=6, theta_max=0, ntheta=4, local_ref_frame=False)
    #myjob.add_thetas(  phi=0, theta_min=0, theta_max=6, ntheta=4, local_ref_frame=False)
    #myjob.add_thetas(phi=270, theta_min=6, theta_max=0, ntheta=4, local_ref_frame=False)
    #myjob.add_thetas( phi=90, theta_min=0, theta_max=6, ntheta=4, local_ref_frame=False)

    # sampling the 3 great circles of on the unit sphere
    # we input the directions in GLOBAL reference frame, so local_ref_frame = False
    myjob.add_thetas(phi=0, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=False)
    myjob.add_thetas(phi=180, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=False)
    myjob.add_thetas(phi=90, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=False)
    myjob.add_thetas(phi=270, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=False)
    myjob.add_phis(theta=90, phi_min=0, phi_max=330, nphi=12, local_ref_frame=False)

    # the only usage I see of the reference energy is in
    # if abs(e - e_ref) < max_energy: proceed as normal
    # type statements
    myjob.e_ref = e_ref
    myjob.print_dirs_and_configs(local_ref_frame=False)

    #### execute these functions one at a time
    ####

    # submit=False creates input files but does not submit jobs so we can manually check
    if(to_do==1):
       myjob.setup_jobs(submit=False);

    # start by running ONLY THE 0th FOLDER to get an existing WAVECAR that the other folders can reference
    elif(to_do==2):
        myjob.setup_jobs(submit=True, truncate_n_directions=1);

    # finally submit all jobs
    elif(to_do==3):
        myjob.setup_jobs(submit=True, truncate_n_directions=1e10);

    # check whether jobs converged and grab final energy of each job -> root_dir/energies.dat
    elif(to_do==4):
        myjob.check_convergences(restart=False, de0=de0)
    elif(to_do==5):
        myjob.get_energies(max_energy=max_energy, de0=de0)
        subprocess.run(["cat", myjob.root_dir+"energies.dat"]);

    else: raise Exception("No action for to_do = {:.0f}".format(to_do));

if __name__ == "__main__":

    control_dir = os.getcwd();
    convergence_threshold = 1.e-8;
    reference_energy = -397.66719288;
    do(int(sys.argv[3]), reference_energy, convergence_threshold); # no idea where these numbers are coming from
    os.chdir(control_dir);
    print("**")
    subprocess.run(["pwd"]);
    subprocess.run(["rm", "data.py"]);
