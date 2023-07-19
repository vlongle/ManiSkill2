import gym
import numpy as np
import open3d as o3d


# http://www.open3d.org/docs/latest/tutorial/Advanced/voxelization.html
# https://towardsdatascience.com/how-to-automate-voxel-modelling-of-3d-point-cloud-with-python-459f4d43a227
# https://www.kaggle.com/code/gzuidhof/reference-model
class ManiSkillVoxelWrapper(gym.ObservationWrapper):
    def __init__(self, env, voxel_size):
        super().__init__(env)
        self.voxel_size = voxel_size

    def observation(self, point_cloud):
        voxels = self.point_cloud_to_voxel(point_cloud)
        return voxels

    def point_cloud_to_voxel(self, point_cloud):
        """
        NOTE: actually not completely understanding this.
        Might just have to use create_from_point_cloud_within_bounds
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)

        # To convert the voxel grid into a numpy array, we can use the get_max_bound and get_min_bound functions 
        # and normalize the voxel to fit into the grid. This will give a binary voxel grid.

        max_bound = voxel_grid.get_max_bound() // self.voxel_size
        min_bound = voxel_grid.get_min_bound() // self.voxel_size
        voxel_size = np.ceil(max_bound - min_bound).astype(int)
        
        voxels = np.zeros(voxel_size)
        for voxel in voxel_grid.get_voxels():
            index = np.floor((voxel.grid_index - min_bound)).astype(int)
            voxels[tuple(index)] = 1

        return voxels


if __name__ == "__main__":

    env = ManiSkillVoxelWrapper(env)
