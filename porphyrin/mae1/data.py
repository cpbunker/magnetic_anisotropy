from os import path
import sys
import subprocess
import numpy as np

# root dir
root_dir = sys.argv[1];
assert(root_dir[-1]=="/");
existing_poscar = root_dir+"POSCAR"
if(path.exists(existing_poscar)): pass
else: raise Exception(existing_poscar+" must already exist!");
existing_pot = root_dir+"POTCAR"
if(path.exists(existing_pot)): pass
else: raise Exception(existing_pot+" must already exist!");
print(">>> pwd");
subprocess.run(["pwd"]);
print(">>> ls "+root_dir);
subprocess.run(["ls", root_dir]);

# wave dir
wave_dir = sys.argv[2];
assert("/" not in wave_dir);

# this string becomes your INCAR file
incar = """ 
SYSTEM = vasp

#### sym ####
#ISYM = 0

#### system size ####
#NBANDS = 448
#NELECT = 383.0

#### accuracy ####
PREC = Accurate
ENCUT = 450
LREAL = F

#### parallelization ####
KPAR = 1
NCORE = 8 # if GPU, leave as is. if CPU, need to change

#### electronic optimization ####
EDIFF = 1E-8
NELM = 90
#NELMIN = 10
#NELMDL = -8
AMIX = 0.3    
AMIX_MAG = 0.6
AMIX_MIN = 0.05
BMIX = 0.0001
BMIX_MAG = 0.0001
ALGO = All

#### structural relaxation ####
#NSW = 1 # static calculation!
#IBRION = 2
#ISIF = 2
#EDIFFG = -0.02
#POTIM = 0.3

#### magnetism: accuracy ####
LASPH = T
GGA_COMPAT = F

#### magnetism: noncollinear spin, SOC #### <--- this is what we are doing
LSORBIT = T
LNONCOLLINEAR = T
SAXIS = {sstring:s}   # <--- this allows us to change this option later in the code
MAGMOM = 0.0 0.0 -1.0 168*0.0 # first vector is the Cu

#### magnetism: constraint ####
I_CONSTRAINED_M = 1 # constrains the spin direction
M_CONSTR =  0.0 0.0 -1.0 168*0.0

LAMBDA =  10.0   # in the penalty term that aligns spin with target spin
RWIGS = 1.164 0.741 0.863 0.370 # 1 number for each element, from POTCARs. Order=Cu, N, C, H

#### magnetism: orbital moment ####
LORBMOM = T

#### charge, wavefunction ####
ISTART = 1
ICHARG = 0
LWAVE = FALSE
LCHARG = F
LAECHG = F
LMAXMIX = 4

#### dos ####
ISMEAR = 0
SIGMA = 0.02
NEDOS = 501
EMIN = -15
EMAX = 10
LORBIT = 11

#### vdW ####
IVDW = 11

#### LDA+U #### recall species are Cu, N, C, H
LDAU = T
LDAUTYPE = 1
LDAUPRINT = 1
LDAUL = 2  -1 -1 -1 # turns on DFT+U for d orbital of copper, all others off
LDAUU = 4.0 0  0  0
LDAUJ = 1.0 0  0  0

#### HSE ####
#LHFCALC = T 
#HFSCREEN = 0.2 
#PRECFOCK = Accurate
#ALGO = All 
#TIME = 0.35

#### wann ####
#LWANNIER90 = .T.
#LWRITE_UNK = .TRUE.

### polarization ###
#EFIELD_PEAD = 0.000 0.000 0.000
#IDIPOL = 3
#LMONO = T
#LDIPOL = T
#LCALCPOL = T
#DIPOL = 0.5 0.5 0.5 
"""

kpoints = """kpoints
0
gamma
1    1   1
 0.0 0.0 0.0
"""

#poscar = """""" # need to copy mine here

# this python code submits a seperate job script for each spin direction

# this uses GPU on perlmutter !!!!
job_script_perlmutter_1_node = """#!/bin/bash
#SBATCH -A m3346_g
#SBATCH -J {dir_name:s}
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -G 4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --error=error
#SBATCH --output=output # don't change since we need to check this later in code
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cpbunker@ufl.edu
##SBATCH --dependency=afterok:45514509

module load vasp/6.4.3-gpu

srun -n 4 -c 32 --cpu-bind=cores -G 4 --gpu-bind=none vasp_ncl
"""

job_script_perlmutter_4_nodes = """#!/bin/bash

#SBATCH -A m3346_g
#SBATCH -J {dir_name:s}
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -G 16
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --error=error
#SBATCH --output=output
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cpbunker@ufl.edu
##SBATCH --dependency=afterok:45514509

module load vasp/6.2.1-gpu

source ~/.bash_aliases; keep_log

srun -n 16 -c 32 --cpu-bind=cores -G 16 --gpu-bind=single:1 vasp_ncl
"""

job_script = job_script_perlmutter_1_node
#print(">>> vasp.job:\n", job_script,"\n>>> End vasp.job\n");

#emats_file = []

#for i in range(100):
#emats_file.append( [[1,0,0],[0,1,0],[0,0,1]] )

# this are the unit vectors that define the reference frame. Since emat = identity, it is just the lab frame
# would be useful to change if we are tackling two porphyrins with different planes simultaneously 
emat_file = np.eye(3);

angle_yax = np.pi/2;
R_yax = np.array([[np.cos(angle_yax), 0          , np.sin(angle_yax)],
                  [0,                 1,           0],
                  [-np.sin(angle_yax), 0,          np.cos(angle_yax)]]);
angle_zax = 2*np.pi/3;
R_zax = np.array([[np.cos(angle_zax),-np.sin(angle_zax),0],
                  [np.sin(angle_zax), np.cos(angle_zax),0],
                  [0,0,1]]);
emat_file = np.transpose( np.matmul(R_zax,np.matmul(R_yax,emat_file.T) ) );


