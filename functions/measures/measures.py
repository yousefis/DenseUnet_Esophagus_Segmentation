import SimpleITK as sitk
import math
import numpy as np

def resampler_sitk(image_sitk, spacing=[1.0, 1.0, 3.0], default_pixel_value=0,
                   interpolator=sitk.sitkNearestNeighbor, dimension=3, rnd=3):
    ratio = [spacing_dim / spacing[i] for i, spacing_dim in enumerate(image_sitk.GetSpacing())]
    ImRef = sitk.Image(tuple(math.ceil(size_dim * ratio[i]) for i, size_dim in enumerate(image_sitk.GetSize())),
                       sitk.sitkInt16)
    ImRef.SetOrigin(image_sitk.GetOrigin())
    ImRef.SetDirection(image_sitk.GetDirection())
    ImRef.SetSpacing(spacing)
    identity = sitk.Transform(dimension, sitk.sitkIdentity)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ImRef)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(identity)
    resampled_sitk = resampler.Execute(image_sitk)

    return resampled_sitk

def DSC_MSD_HD95(groundtruth_image_itk, predicted_image, resample_flag=False, resample_spacing=[1.0, 1.0, 3.0]):

    if resample_flag:
        groundtruth_image_itk = resampler_sitk(image_sitk=groundtruth_image_itk, spacing=resample_spacing,
                                                default_pixel_value=0,
                                                interpolator=sitk.sitkNearestNeighbor, dimension=3, rnd=3)

    groundtruth_image_itk = sitk.Cast(groundtruth_image_itk, sitk.sitkUInt8)
    predicted_image = sitk.Cast(predicted_image, sitk.sitkUInt8)
    size_diff = np.sum(np.subtract(groundtruth_image_itk.GetSize(), predicted_image.GetSize()))

    if size_diff > 0:
        if size_diff == 2:
            groundtruth_image_itk = groundtruth_image_itk[:-1, :-1, :]
        elif size_diff == 2:
            groundtruth_image_itk = groundtruth_image_itk[:-1, :-1, :-1]
        else:
            print(size_diff)

    elif size_diff < 0:
        if size_diff == -2:
            predicted_image = predicted_image[:-1, :-1, :]
        elif size_diff == -3:
            predicted_image = predicted_image[:-1, :-1, :-1]
        else:
            print(size_diff)

    else:
        pass

    label_overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    try:
        label_overlap_measures_filter.Execute(groundtruth_image_itk, predicted_image)

        dice = label_overlap_measures_filter.GetDiceCoefficient()
        jaccard = label_overlap_measures_filter.GetJaccardCoefficient()
        vol_similarity = label_overlap_measures_filter.GetVolumeSimilarity()
    except:
        dice =-1
        jaccard =-1
        vol_similarity =-1
    try:
        hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_image_filter.Execute(groundtruth_image_itk, predicted_image)
    except:
        return -1, -1, -1, -1, -1

    reference_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(groundtruth_image_itk, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(groundtruth_image_itk)

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    segmented_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(predicted_image, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(predicted_image)

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))

    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

    all_surface_distances = seg2ref_distances + ref2seg_distances
    msd = np.mean(all_surface_distances)
    hd_percentile = np.maximum(np.percentile(seg2ref_distances, 95), np.percentile(ref2seg_distances, 95))

    return dice, msd, hd_percentile, jaccard, vol_similarity