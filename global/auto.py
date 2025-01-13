import os
import subprocess

control_dir = os.getcwd();
subprocess.run(['control_dir="',control_dir,'"']);
subprocess.run(["echo", "$control_dir"]);
