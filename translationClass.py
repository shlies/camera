import numpy as np
import cv2

class Translation:
    def __init__(self):
        print("Translation started successfully!")

    fx = 545.103
    fy = 545.103
    cx = 321.608
    cy = 243.754
    k1 = 0.0887278
    k2 = -0.0544192
    k3 = -0.297869
    k4 = 0
    k5 = 0
    k6 = 0
    p1 = -0.000272924
    p2 = 0.00631082

    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rotation_vector = np.array([0, 0, 0])
    translation_vector = np.array([0, 0, 0])
    rgb_distortion_matrix = np.array([[k1,k2,p1],[k3,k4,p2],[k5,k6,1]])


    def image_to_world_vectorized(self, coords, depth_map, M, r_vec, T, rgb_distortion_matrix):
        u, v = coords[0], coords[1]
        z = depth_map[v, u]  # Extract depth values for each coordinate
        i_vector = np.stack([u, v, np.ones_like(u)], axis=-1)  # Convert to homogeneous coordinates

        # Apply RGB distortion matrix
        i_vector_distorted = np.matmul(rgb_distortion_matrix, i_vector.T).T[..., :2]

        i_vector_transformed = i_vector_distorted @ np.linalg.inv(M).T
        rot_mat, _ = cv2.Rodrigues(r_vec)
        rot_mat_inv = np.linalg.inv(rot_mat)

        left_side = rot_mat_inv @ i_vector_transformed[..., np.newaxis]  # Shape: [height, width, 3, 1]

        # Ensure 'right_side' is broadcastable with 'left_side'
        right_side = rot_mat_inv @ T.reshape(-1, 1)  # Shape: [3, 1]
        right_side = right_side.reshape(1, 1, 3, 1)  # Reshaped to [1, 1, 3, 1]

        # Reshape 's' for broadcasting
        s = z[..., np.newaxis, np.newaxis]  # Shape becomes [height, width, 1, 1]
        w = s * left_side - right_side
        w = w.squeeze(-1)  # Remove the last dimension, shape: [height, width, 3]

        return w

    def show_z_channel(self, world_coordinates):
        # Extract the Z channel
        z_channel = world_coordinates[:, :, 2]

        # Normalize the Z channel for better visualization
        z_norm = cv2.normalize(z_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Display the Z channel
        cv2.imshow('Z Channel', z_norm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def trans(self, coords_in, depth_map_in,):
        self.world_coord = self.image_to_world_vectorized(coords_in, depth_map_in, self.intrinsic_matrix, self.rotation_vector, self.translation_vector, self.rgb_distortion_matrix)
