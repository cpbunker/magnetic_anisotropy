import os
import numpy as np
from common import *

def find_global_directions():
    direction_emin = (71.4,  59.5)
    myjob = vasp_jobs_ncl(n_theta=1801, n_phi=3600)
    myjob.setup_local_ref_frames(z_direction=direction_emin, from_file=False)
    myjob.add_directions([[0.92, 180]], local_ref_frame=True)
    myjob.print_dirs_and_configs(local_ref_frame=False)
    #print("Opposite direction: {:6.1f} {:6.1f}".format(*opposite_direction))

def do(to_do, e_ref, de0, direction_emin = (0.0, 0.0), max_energy = np.inf):

    myjob = vasp_jobs_ncl(n_theta=1801, n_phi=3600)
    #myjob.setup_local_ref_frames(z_direction=direction_emin, from_file=True)

    #myjob.add_directions([[0,0]], local_ref_frame=False)

    #myjob.add_thetas(  phi=0, theta_min=0, theta_max=30, ntheta=4, local_ref_frame=True)

    #myjob.add_thetas(phi=0, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=True)
    #myjob.add_thetas(phi=180, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=True)
    #myjob.add_thetas(phi=90, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=True)
    #myjob.add_thetas(phi=270, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=True)
    #myjob.add_phis(theta=90, phi_min=0, phi_max=330, nphi=12, local_ref_frame=True)

    #myjob.add_phis(theta=90, phi_min=315, phi_max=345, nphi=7, local_ref_frame=True)
    #myjob.add_thetas(phi=330, theta_min=75, theta_max=105, ntheta=7, local_ref_frame=True)

    #myjob.setup_local_ref_frames(z_direction=direction_emin, from_file=False)
    #myjob.add_thetas(phi=180, theta_min=6, theta_max=0, ntheta=4, local_ref_frame=True)
    #myjob.add_thetas(  phi=0, theta_min=0, theta_max=6, ntheta=4, local_ref_frame=True)
    #myjob.add_thetas(phi=270, theta_min=6, theta_max=0, ntheta=4, local_ref_frame=True)
    #myjob.add_thetas( phi=90, theta_min=0, theta_max=6, ntheta=4, local_ref_frame=True)

    # sampling the 3 great circles of on the unit sphere
    myjob.add_thetas(phi=0, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=True)
    myjob.add_thetas(phi=180, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=True)
    myjob.add_thetas(phi=90, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=True)
    myjob.add_thetas(phi=270, theta_min=0, theta_max=180, ntheta=7, local_ref_frame=True)
    myjob.add_phis(theta=90, phi_min=0, phi_max=330, nphi=12, local_ref_frame=True)

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
        myjob.get_energies(max_energy=max_energy, de0=de0)

    else: raise Exception("No action for to_do = {:.0f}".format(to_do));

if __name__ == "__main__":

    convergence_threshhold = 1.e-8;
    reference_energy = -397.66719288;
    do(sys.argv[3], reference_energy, convergence_threshold, direction_emin = (98.282, 330.142)); # no idea where these numbers are coming from
