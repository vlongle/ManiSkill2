# parse .h5
import h5py

file_path = "demos/rigid_body/LiftCube-v0/trajectory.h5"
f = h5py.File(file_path, 'r')
traj = f["traj_0"]
print(traj["actions"])
print(traj["success"])
print(traj["env_states"])