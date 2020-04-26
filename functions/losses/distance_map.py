import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from scipy.spatial import distance_matrix
# from sklearn.metrics import pairwise_distances_chunked
from functions.read_data2 import _read_data






def distance(path):
    # Make input data
    # w, h, d = 10, 20, 30
    # t = tf.cast(tf.random_uniform([w, h, d], 0,10)>7, tf.int32)
    # print(t.numpy())
    # [[[1 1 1 1]
    #   [1 1 1 1]
    #   [1 1 0 0]]
    #
    #  [[1 1 1 1]
    #   [1 1 1 1]
    #   [1 1 1 1]]]
    # Find coordinates that are positive and on the surface
    # (surrounded but at least one 0)
    I = sitk.ReadImage(path)
    II = sitk.GetArrayFromImage(I)
    dis_map = np.ones(np.shape(II))
    nonzero = np.where(II)
    nonzero_minx = np.min(nonzero[0])
    nonzero_maxx = np.max(nonzero[0])
    nonzero_miny = np.min(nonzero[1])
    nonzero_maxy = np.max(nonzero[1])
    nonzero_minz = np.min(nonzero[2])
    nonzero_maxz = np.max(nonzero[2])
    r = 10
    inp=II[nonzero_minx - r:nonzero_maxx + r,nonzero_miny - r:nonzero_maxy + r, nonzero_minz - r:nonzero_maxz + r]

    splits = str.rsplit(path, 'GTV')
    dismap_name=(splits[0] + 'distancemap' + splits[1])
    t=tf.convert_to_tensor(inp)
    w,h,d=np.shape(inp)
    t_pad_z = tf.pad(t, [(1, 1), (1, 1), (1, 1)]) <= 0
    m_pos = t > 0
    m_surround_z = tf.zeros_like(m_pos)
    # Go through the 6 surrounding positions
    for i in range(3):
        for s in [slice(None, -2), slice(2, None)]:
            slices = tuple(slice(1, -1) if i != j else s for j in range(3))
            m_surround_z |= t_pad_z.__getitem__(slices)
    # Surface points are positive points surrounded by some zero
    m_surf = m_pos & m_surround_z
    coords_surf = tf.where(m_surf)
    # Following from before
    # coords_surf = ...
    coords_z = tf.where(~m_pos)

    CHUNK_SIZE = 1_000 # Choose chunk size
    dtype = tf.float32
    # If using TF 2.x you can know in advance the size of the tensor array
    # (although the element shape will not be constant due to the last chunk)
    num_z = tf.shape(coords_z)[0]
    arr = tf.TensorArray(dtype, size=(num_z - 1) // CHUNK_SIZE + 1, element_shape=[None], infer_shape=False)
    _, arr = tf.while_loop(lambda i, arr: i < num_z,
                           lambda i, arr: (i + CHUNK_SIZE, arr.write(i // CHUNK_SIZE,
                               tf.reduce_min(tf.linalg.norm(tf.cast(
                                   tf.reshape(coords_z[i:i + CHUNK_SIZE], [-1, 1, 3]) - coords_surf,
                               dtype), axis=-1), axis=-1))),
                           [tf.constant(0, tf.int32), arr])
    min_dists = arr.concat()
    out = tf.scatter_nd(coords_z, min_dists, [w, h, d])
    sess = tf.Session()
    result = sess.run(out)

    result = result / np.max(result)
    dis_map[nonzero_minx - r:nonzero_maxx + r,
    nonzero_miny - r:nonzero_maxy + r,
    nonzero_minz - r:nonzero_maxz + r] = result

    sitk_dismap=sitk.GetImageFromArray(dis_map.astype(np.float32))
    sitk_dismap.SetOrigin(I.GetOrigin())
    sitk_dismap.SetDirection(I.GetDirection())
    sitk_dismap.SetSpacing(I.GetSpacing())
    sitk.WriteImage(sitk_dismap,dismap_name)
    print(dismap_name)
    return dis_map

# def distance_penalizing_generator():


if __name__=='__main__':
    # train_tag = '',
    # validation_tag = '',
    # test_tag = '',
    # img_name = ''
    # label_name = ''
    # torso_tag = '',
    # tumor_percent = .75,
    # other_percent = .25,
    fold = 0

    _rd = _read_data(data=2,
                     train_tag='',
                     validation_tag='',
                     test_tag='',
                     img_name='',
                     label_name='',
                     torso_tag='')
    train_CTs, train_GTVs, train_Torso, train_penalize, \
    validation_CTs, validation_GTVs, validation_Torso, validation_penalize, \
    test_CTs, test_GTVs, test_Torso, test_penalize = _rd.read_data_path(fold=fold)
    # path = '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v4/zz1010380882/zz1010380882/GTV_re113.mha'
    for train in train_GTVs:
        dis_map=distance(train)

    for val in validation_GTVs:
        dis_map=distance(val)

    for test in test_GTVs:
        dis_map=distance(test)


