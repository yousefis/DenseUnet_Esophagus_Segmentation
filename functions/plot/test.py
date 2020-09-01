import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Make input data
w, h, d = 10, 20, 30
w, h, d = 2, 3, 5
t= tf.placeholder(shape=[w,h,d],dtype=tf.int8)

# Make coordinates
coords = tf.meshgrid(tf.range(w), tf.range(h), tf.range(d), indexing='ij')
coords = tf.stack(coords, axis=-1)
# Find coordinates that are positive
m = t > 0
coords_pos = tf.boolean_mask(coords, m)
# Find every pairwise distance
vec_d = tf.reshape(coords, [-1, 1, 3]) - coords_pos
# You may choose a difference precision type here
dists = tf.linalg.norm(tf.dtypes.cast(vec_d, tf.float32), axis=-1)
# Find minimum distances
min_dists = tf.reduce_min(dists, axis=-1)
# Reshape
out = tf.reshape(min_dists, [w, h, d])

sess=tf.Session()
T=np.array([[[0 ,1 ,0 ,0,0],
            [0 ,1 ,1 ,0,0],
            [0 ,1 ,1 ,0,0]],
           [[0 ,0 ,0 ,0,0],
            [0 ,0 ,0 ,0,0],
            [0 ,0 ,0 ,0,0]]])
[o]=sess.run([out], feed_dict={t:T})

plt.subplot(1,2,1)
plt.imshow(T[:,:,4],cmap='jet')
plt.subplot(1,2,2)
plt.imshow(o[:,:,4],cmap='jet')

plt.show()
# print(out.numpy().round(3))
# [[[1.    0.    1.    2.   ]
#   [1.    1.    1.414 1.   ]
#   [0.    0.    1.    0.   ]]
#
#  [[0.    1.    1.414 2.236]
#   [1.    1.    1.414 1.414]
#   [0.    0.    1.    1.   ]]]
#
# import tensorflow as tf
#
# # Make input data
# w, h, d = 10, 20, 30
# w, h, d = 2, 3, 4
# t = tf.dtypes.cast(tf.random.stateless_uniform([w, h, d], (0, 0)) > .15, tf.int32)
# print(t.numpy())
# # [[[1 1 1 1]
# #   [1 1 1 1]
# #   [1 1 0 0]]
# #
# #  [[1 1 1 1]
# #   [1 1 1 1]
# #   [1 1 1 1]]]
# # Find coordinates that are positive and on the surface
# # (surrounded but at least one 0)
# t_pad_z = tf.pad(t, [(1, 1), (1, 1), (1, 1)]) <= 0
# m_pos = t > 0
# m_surround_z = tf.zeros_like(m_pos)
# # Go through the 6 surrounding positions
# for i in range(3):
#     for s in [slice(None, -2), slice(2, None)]:
#         slices = tuple(slice(1, -1) if i != j else s for j in range(3))
#         m_surround_z |= t_pad_z.__getitem__(slices)
# # Surface points are positive points surrounded by some zero
# m_surf = m_pos & m_surround_z
# coords_surf = tf.where(m_surf)
# # Find coordinates that are zero
# coords_z = tf.where(~m_pos)
# # Find every pairwise distance
# vec_d = tf.reshape(coords_z, [-1, 1, 3]) - coords_surf
# dists = tf.linalg.norm(tf.dtypes.cast(vec_d, tf.float32), axis=-1)
# # Find minimum distances
# min_dists = tf.reduce_min(dists, axis=-1)
# # Put minimum distances in output array
# out = tf.scatter_nd(coords_z, min_dists, [w, h, d])
# print(out.numpy().round(3))
# [[[0. 0. 0. 0.]
#   [0. 0. 0. 0.]
#   [0. 0. 1. 1.]]
#
#  [[0. 0. 0. 0.]
#   [0. 0. 0. 0.]
#   [0. 0. 0. 0.]]]

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
#
# def get_all_boundary_edges(bool_img):
#     """
#     Get a list of all edges
#     (where the value changes from 'True' to 'False') in the 2D image.
#     Return the list as indices of the image.
#     """
#     ij_boundary = []
#     ii, jj = np.nonzero(bool_img)
#     for i, j in zip(ii, jj):
#         # North
#         if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
#             ij_boundary.append(np.array([[i, j+1],
#                                          [i+1, j+1]]))
#         # East
#         if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
#             ij_boundary.append(np.array([[i+1, j],
#                                          [i+1, j+1]]))
#         # South
#         if j == 0 or not bool_img[i, j-1]:
#             ij_boundary.append(np.array([[i, j],
#                                          [i+1, j]]))
#         # West
#         if i == 0 or not bool_img[i-1, j]:
#             ij_boundary.append(np.array([[i, j],
#                                          [i, j+1]]))
#     if not ij_boundary:
#         return np.zeros((0, 2, 2))
#     else:
#         return np.array(ij_boundary)
#
#
#
#
# def close_loop_boundary_edges(xy_boundary, clean=True):
#     """
#     Connect all edges defined by 'xy_boundary' to closed
#     boundary lines.
#     If not all edges are part of one surface return a list of closed
#     boundaries is returned (one for every object).
#     """
#
#     boundary_loop_list = []
#     while xy_boundary.size != 0:
#         # Current loop
#         xy_cl = [xy_boundary[0, 0], xy_boundary[0, 1]]  # Start with first edge
#         xy_boundary = np.delete(xy_boundary, 0, axis=0)
#
#         while xy_boundary.size != 0:
#             # Get next boundary edge (edge with common node)
#             ij = np.nonzero((xy_boundary == xy_cl[-1]).all(axis=2))
#             if ij[0].size > 0:
#                 i = ij[0][0]
#                 j = ij[1][0]
#             else:
#                 xy_cl.append(xy_cl[0])
#                 break
#
#             xy_cl.append(xy_boundary[i, (j + 1) % 2, :])
#             xy_boundary = np.delete(xy_boundary, i, axis=0)
#
#         xy_cl = np.array(xy_cl)
#
#         boundary_loop_list.append(xy_cl)
#
#     return boundary_loop_list
#
# def plot_world_outlines(bool_img, ax=None, **kwargs):
#     if ax is None:
#         ax = plt.gca()
#
#     ij_boundary = get_all_boundary_edges(bool_img=bool_img)
#     xy_boundary = ij_boundary - 0.5
#     xy_boundary = close_loop_boundary_edges(xy_boundary=xy_boundary)
#     cl = LineCollection(xy_boundary, **kwargs)
#     ax.add_collection(cl)
#
#
# array = np.zeros((20, 20))
# array[4:7, 3:8] = 1
# array[4:7, 12:15] = 1
# array[7:15, 7:15] = 1
# array[12:14, 13:14] = 0
#
# plt.figure()
# plt.imshow(array, cmap='binary')
# plot_world_outlines(array.T, lw=5, color='r')
