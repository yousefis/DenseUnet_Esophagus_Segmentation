import numpy as np
import SimpleITK as sitk
import collections
from random import shuffle
import functions.settings as settings
import random
import itertools

class image_class:
    def __init__(self,CTs,GTVs,Torsos,Penalize
                 ,bunch_of_images_no=20,is_training=1,patch_window=47,gtv_patch_window=33):
        self.CTs=CTs
        self.GTVs=GTVs
        self.Torsos=Torsos
        self.Penalize=Penalize
        self.bunch_of_images_no = bunch_of_images_no
        self.node = collections.namedtuple('node', 'img_index CT_image GTV_image Torso_image  Penalize_image min_torso max_torso voxel_size origin direction depth width height ct_name gtv_name torso_name penalize_name')
        self.collection=[]
        self.is_training=is_training
        self.patch_window=patch_window
        self.gtv_patch_window=gtv_patch_window
        self.random_images=list(range(0,len(self.CTs)))
        self.min_image=-1024
        self.max_image=1500
        self.counter_save=0
        self.end_1st_dataset=375
        self.random_data1 = list(range(0, self.end_1st_dataset))
        self.random_data2 = list(range(self.end_1st_dataset, 456))
        self.deform1stdb=0
        self.static_counter_vl=0
        self.seed = 100




    # --------------------------------------------------------------------------------------------------------
    def creat_mask(self,shape):
        torso_mask = np.ones((shape[0] - int(self.patch_window),
                               shape[1] - int(self.patch_window),
                               shape[2] - int(self.patch_window)))

        torso_mask = np.pad(torso_mask, (
            (int(self.patch_window / 2) + 1, int(self.patch_window / 2)),
            (int(self.patch_window / 2) + 1, int(self.patch_window / 2)),
            (int(self.patch_window / 2) + 1, int(self.patch_window / 2)),
        ),
                                  mode='constant', constant_values=0)
        return torso_mask


    # --------------------------------------------------------------------------------------------------------
    def image_padding(self,img, padLowerBound, padUpperBound, constant):
        filt = sitk.ConstantPadImageFilter()
        padded_img = filt.Execute(img,
                                  padLowerBound,
                                  padUpperBound,
                                  constant)
        return padded_img

    def image_crop(self,img, padLowerBound, padUpperBound):
        crop_filt = sitk.CropImageFilter()
        cropped_img = crop_filt.Execute(img, padLowerBound, padUpperBound)
        return cropped_img

    def apply_deformation(self,img, BCoeff, defaultPixelValue, spacing, origin, direction, interpolator):
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(BCoeff)
        resampler.SetDefaultPixelValue(defaultPixelValue)
        resampler.SetReferenceImage(img)  # set input image
        resampler.SetInterpolator(interpolator)  # set interpolation method
        resampler.SetOutputSpacing(spacing)
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputDirection(direction)
        deformedImg = sitk.Resample(img, BCoeff)
        return deformedImg

    def Bspline_distort(self,CT_image, GTV_image, Torso_image, Penalize_image,max_dis=2):
        z_len = CT_image.GetDepth()
        x_len = CT_image.GetHeight()
        gen = self.random_gen(1, max_dis)
        displace_range = list(itertools.islice(gen, 1))[0]
        grid_space = 4
        z_grid=0
        while not z_grid:
            grid_space+=1
            z_grid = int(grid_space * z_len / (x_len))



        spacing = CT_image.GetSpacing()
        origin = CT_image.GetOrigin()
        direction = CT_image.GetDirection()

        BCoeff = sitk.BSplineTransformInitializer(CT_image, [grid_space, grid_space, z_grid],
                                                  order=3)
        # The third parameter for the BSplineTransformInitializer is the spline order It defaults to 3
        displacements = np.random.uniform(-displace_range, displace_range, int(len(BCoeff.GetParameters())))

        Xdisplacements = np.reshape(displacements[0: (grid_space + 3) * (grid_space + 3) * (z_grid + 3)],
                                    [(grid_space + 3), (grid_space + 3), (z_grid + 3)])
        Ydisplacements = np.reshape(displacements[
                                    (grid_space + 3) * (grid_space + 3) * (z_grid + 3): 2 * (grid_space + 3) * (
                                    grid_space + 3) * (z_grid + 3)],
                                    [(grid_space + 3), (grid_space + 3), (z_grid + 3)])
        Zdisplacements = np.reshape(displacements[
                                    2 * (grid_space + 3) * (grid_space + 3) * (z_grid + 3):3 * (grid_space + 3) * (
                                    grid_space + 3) * (z_grid + 3)],
                                    [(grid_space + 3), (grid_space + 3), (z_grid + 3)])

        displacements = np.hstack((np.reshape(Xdisplacements, -1),
                                   np.reshape(Ydisplacements, -1),
                                   np.reshape(Zdisplacements, -1)))
        BCoeff.SetParameters(displacements)

        # define sampler
        CT_deformed = self.apply_deformation(img=CT_image, BCoeff=BCoeff,
                                        defaultPixelValue=-1024, spacing=spacing,
                                        origin=origin, direction=direction,
                                        interpolator=sitk.sitkBSpline)
        # define sampler for gtv

        GTV_deformed = self.apply_deformation(img=GTV_image, BCoeff=BCoeff,
                                         defaultPixelValue=0, spacing=spacing,
                                         origin=origin, direction=direction,
                                         interpolator=sitk.sitkNearestNeighbor)


        Torso_deformed = self.apply_deformation(img=Torso_image, BCoeff=BCoeff,
                                           defaultPixelValue=0, spacing=spacing,
                                           origin=origin, direction=direction,
                                           interpolator=sitk.sitkNearestNeighbor)


        return CT_deformed, GTV_deformed, Torso_deformed, []
    def Bspline_distort2(self,CT_image, GTV_image, Torso_image, Penalize_image):
        grid_space = 2
        gen = self.random_gen(1, 5)
        displace_range = list(itertools.islice(gen, 1))[0]


        spacing = CT_image.GetSpacing()
        origin = CT_image.GetOrigin()
        direction = CT_image.GetDirection()
        z_len = CT_image.GetDepth()
        x_len = CT_image.GetHeight()
        padd_zero = x_len - z_len

        CT_image1 = self.image_padding(img=CT_image,
                                  padLowerBound=[0, 0, int(padd_zero / 2)],
                                  padUpperBound=[0, 0, int(padd_zero / 2)],
                                  constant=-1024)

        GTV_image1 = self.image_padding(img=GTV_image,
                                   padLowerBound=[0, 0, int(padd_zero / 2)],
                                   padUpperBound=[0, 0, int(padd_zero / 2)],
                                   constant=-1024)

        Torso_image1 = self.image_padding(img=Torso_image,
                                     padLowerBound=[0, 0, int(padd_zero / 2)],
                                     padUpperBound=[0, 0, int(padd_zero / 2)],
                                     constant=-1024)

        # CT_image2 = image_crop(CT_image1, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])
        # GTV_image2 = image_crop(GTV_image1, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])
        # Torso_image2 = image_crop(Torso_image1, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])
        #
        # sitk.WriteImage(CT_image2, '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/test/ct2.mha')
        # sitk.WriteImage(GTV_image2, '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/test/gtv2.mha')
        # sitk.WriteImage(Torso_image2, '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/test/torso2.mha')

        # define transform:
        BCoeff = sitk.BSplineTransformInitializer(CT_image1, [grid_space, grid_space, grid_space],
                                                  order=3)
        # The third parameter for the BSplineTransformInitializer is the spline order It defaults to 3
        displacements = np.random.uniform(-displace_range, displace_range, int(len(BCoeff.GetParameters())))
        param_no = np.int(np.ceil(np.power(len(displacements) / 3, 1 / 3)))

        Xdisplacements = np.reshape(displacements[0: param_no * param_no * param_no],
                                    [param_no, param_no, param_no])
        Ydisplacements = np.reshape(
            displacements[param_no * param_no * param_no: 2 * param_no * param_no * param_no],
            [param_no, param_no, param_no])
        Zdisplacements = np.reshape(
            displacements[2 * param_no * param_no * param_no:3 * param_no * param_no * param_no],
            [param_no, param_no, param_no])

        displacements = np.hstack((np.reshape(Xdisplacements, -1),
                                   np.reshape(Ydisplacements, -1),
                                   np.reshape(Zdisplacements, -1)))
        BCoeff.SetParameters(displacements)

        # define sampler
        CT_deformed = self.apply_deformation(img=CT_image1, BCoeff=BCoeff,
                                        defaultPixelValue=-1024, spacing=spacing,
                                        origin=origin, direction=direction,
                                        interpolator=sitk.sitkBSpline)
        CT_deformed = self.image_crop(CT_deformed, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])

        GTV_deformed = self.apply_deformation(img=GTV_image1, BCoeff=BCoeff,
                                         defaultPixelValue=0, spacing=spacing,
                                         origin=origin, direction=direction,
                                         interpolator=sitk.sitkNearestNeighbor)
        GTV_deformed = self.image_crop(GTV_deformed, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])

        Torso_deformed = self.apply_deformation(img=Torso_image1, BCoeff=BCoeff,
                                           defaultPixelValue=0, spacing=spacing,
                                           origin=origin, direction=direction,
                                           interpolator=sitk.sitkNearestNeighbor)
        Torso_deformed = self.image_crop(Torso_deformed, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])


        return CT_deformed, GTV_deformed, Torso_deformed, []


    # --------------------------------------------------------------------------------------------------------
    def Flip(self,CT_image, GTV_image, Torso_image):
        TF1=False
        TF2=bool(random.getrandbits(1))
        TF3=bool(random.getrandbits(1))

        CT_image = sitk.Flip(CT_image, [TF1, TF2, TF3])
        GTV_image = sitk.Flip(GTV_image, [TF1, TF2, TF3])
        Torso_image = sitk.Flip(Torso_image, [TF1, TF2, TF3])
        return CT_image, GTV_image, Torso_image
    # --------------------------------------------------------------------------------------------------------
    def HistogramEqualizer(self,CT_image1):
        return CT_image1

    # --------------------------------------------------------------------------------------------------------
    # read information of each image
    def read_unpadded_image(self, img_index, deform, max_dis=0): #read images which are not padded. it is a bit slow
        # print(self.CTs[int(img_index)])
        bth = sitk.BinaryThresholdImageFilter()
        CT_image1 = sitk.ReadImage(''.join(self.CTs[int(img_index)]))
        voxel_size = CT_image1.GetSpacing()
        origin = CT_image1.GetOrigin()
        direction = CT_image1.GetDirection()

        GTV_image1 = sitk.ReadImage(''.join(self.GTVs[int(img_index)]))
        Torso_image1 = sitk.ReadImage(''.join(self.Torsos[int(img_index)]))
        Penalize_image1 = []  # sitk.ReadImage(''.join(self.Penalize[int(img_index)]))

        padding_size = self.patch_window + 15  # 87

        CT_image1 = self.image_padding(img=CT_image1,
                                       padLowerBound=[int(padding_size / 2), int(padding_size / 2),
                                                      int(padding_size / 2)],
                                       padUpperBound=[int(padding_size / 2) + 1, int(padding_size / 2) + 1,
                                                      int(padding_size / 2) + 1],
                                       constant=-1024)

        max_filt = sitk.MinimumMaximumImageFilter()
        max_filt.Execute(GTV_image1)
        max_val = max_filt.GetMaximum()
        if max_val == 255:
            GTV_image1 = bth.Execute(GTV_image1, 255, 255, 1, 0)

        GTV_image1 = self.image_padding(img=GTV_image1,
                                        padLowerBound=[int(padding_size / 2), int(padding_size / 2),
                                                       int(padding_size / 2)],
                                        padUpperBound=[int(padding_size / 2) + 1, int(padding_size / 2) + 1,
                                                       int(padding_size / 2) + 1],
                                        constant=0)

        Torso_image1 = self.image_padding(img=Torso_image1,
                                          padLowerBound=[int(padding_size / 2), int(padding_size / 2),
                                                         int(padding_size / 2)],
                                          padUpperBound=[int(padding_size / 2) + 1, int(padding_size / 2) + 1,
                                                         int(padding_size / 2) + 1], constant=0)

        if deform == 1:
            [CT_image1, GTV_image1, Torso_image1, Penalize_image1] = self.Bspline_distort(CT_image1, GTV_image1,
                                                                                          Torso_image1,
                                                                                          Penalize_image1,
                                                                                          max_dis=max_dis)

        CT_image = sitk.GetArrayFromImage(CT_image1)
        GTV_image = sitk.GetArrayFromImage(GTV_image1)
        Torso_image_mul = sitk.GetArrayFromImage(Torso_image1)

        ct_name = self.CTs[int(img_index)]
        gtv_name = self.GTVs[int(img_index)]
        torso_name = self.Torsos[int(img_index)]

        min_torso = np.min(np.where(Torso_image_mul == 1)[0])
        max_torso = np.max(np.where(Torso_image_mul == 1)[0])

        depth = GTV_image.shape[0]
        width = GTV_image.shape[1]
        height = GTV_image.shape[2]

        n = self.node(img_index=img_index, CT_image=CT_image, GTV_image=GTV_image, Torso_image=Torso_image_mul,
                      Penalize_image=[],
                      min_torso=min_torso, max_torso=max_torso,
                      voxel_size=voxel_size, origin=origin, direction=direction,
                      depth=depth, width=width, height=height, ct_name=ct_name, gtv_name=gtv_name,
                      torso_name=torso_name)
        return n
    # --------------------------------------------------------------------------------------------------------
    #read information of each image
    def read_image(self,img_index,deform,max_dis=0): #read padded images
        bth = sitk.BinaryThresholdImageFilter()
        CT_image1 = sitk.ReadImage(''.join(self.CTs[int(img_index)]))
        voxel_size = CT_image1.GetSpacing()
        origin = CT_image1.GetOrigin()
        direction = CT_image1.GetDirection()

        GTV_image1 = sitk.ReadImage(''.join(self.GTVs[int(img_index)]))
        Torso_image1 = sitk.ReadImage(''.join(self.Torsos[int(img_index)]))
        Penalize_image1 =sitk.ReadImage(''.join(self.Penalize[int(img_index)]))


        max_filt=sitk.MinimumMaximumImageFilter()
        max_filt.Execute(GTV_image1)
        max_val= max_filt.GetMaximum()
        if max_val==255:
            GTV_image1 = bth.Execute(GTV_image1, 255, 255, 1, 0)


        if deform==1 :
            [CT_image1,GTV_image1,Torso_image1,Penalize_image1]=self.Bspline_distort(CT_image1,GTV_image1,Torso_image1,Penalize_image1,max_dis=max_dis)


        CT_image = sitk.GetArrayFromImage(CT_image1)
        GTV_image = sitk.GetArrayFromImage(GTV_image1)
        Torso_image_mul = sitk.GetArrayFromImage(Torso_image1)
        Penalize_image = sitk.GetArrayFromImage(Penalize_image1)

        ct_name=self.CTs[int(img_index)]
        gtv_name=self.GTVs[int(img_index)]
        torso_name=self.Torsos[int(img_index)]
        penalize_name = self.Penalize[int(img_index)]


        min_torso = np.min(np.where(Torso_image_mul == 1)[0])
        max_torso = np.max(np.where(Torso_image_mul == 1)[0])
        depth = GTV_image.shape[0]
        width=GTV_image.shape[1]
        height = GTV_image.shape[2]
        n = self.node(img_index=img_index, CT_image=CT_image, GTV_image=GTV_image, Torso_image=Torso_image_mul,Penalize_image=Penalize_image,
                      min_torso=min_torso,max_torso=max_torso,
                      voxel_size=voxel_size, origin=origin, direction=direction,
                      depth=depth, width=width,height=height,ct_name=ct_name,
                      gtv_name=gtv_name, torso_name=torso_name,penalize_name=penalize_name)
        return n

    def return_normal_image(self,CT_image,max_range,min_range,min_normal,max_normal):
        return (max_range - min_range) * (
        (CT_image - min_normal) / (max_normal - min_normal)) + min_range
    # --------------------------------------------------------------------------------------------------------
    def random_gen(self,low, high):
        while True:
            yield random.randrange(low, high)
    # --------------------------------------------------------------------------------------------------------
    def read_bunch_of_images_from_all_datasets(self):  # for training
        if settings.tr_isread==False:
            return
        settings.read_patche_mutex_tr.acquire()
        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)

        if len(self.random_images) < self.bunch_of_images_no:  # if there are no image choices for selection
            self.random_images = list(range(0, len(self.CTs)))
            self.deform1stdb+=1
            settings.epochs_no+=1
        # select some distinct images for extracting patches!
        distinct_patient_1st_dataset = np.random.randint(0, 13, int(self.bunch_of_images_no/3))#max:14#select ? distinct patients from the 1st dataset
        scan_1st_dataset = np.random.randint(0, 26, int(self.bunch_of_images_no/3)) #max:24#select ? numbers for determining which scan
        rand_image_no1 = distinct_patient_1st_dataset*24+scan_1st_dataset
        rand_image_no_2=np.random.randint(300, 395, int(self.bunch_of_images_no/3)) #select ? patient from the second dataset
        rand_image_no_3=np.random.randint(395, len(self.CTs), int(self.bunch_of_images_no/3)) #select ? patient from the third dataset

        rand_image_no=np.hstack((rand_image_no1, rand_image_no_2))
        rand_image_no=np.hstack((rand_image_no, rand_image_no_3))


        self.random_images = [x for x in range(len(self.random_images)) if
                              x not in rand_image_no]  # remove selected images from the choice list
        print(rand_image_no)

        for img_index in range(len(rand_image_no)):
            deform_int = random.uniform(0, 1)
            deform = 0
            max_dis=0



            imm = self.read_image(rand_image_no[img_index], deform=deform,max_dis=max_dis)
            if len(imm) == 0:

                continue

            self.collection.append(imm)
            print('train image no read so far: %s'%len(self.collection))

        settings.tr_isread=False
        settings.read_patche_mutex_tr.release()

    def read_bunch_of_images_from2nd_dataset(self):  # for training
        if settings.tr_isread==False:
            return
        settings.read_patche_mutex_tr.acquire()

        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)

        if len(self.random_images) < self.bunch_of_images_no:  # if there are no image choices for selection
            self.random_images = list(range(0, len(self.CTs)))
            self.deform1stdb+=1
        # select some distinct images for extracting patches!

        rand_image_no_2=np.random.randint(0, len(self.CTs), int(self.bunch_of_images_no)) #select 24 patient from the second dataset
        print(rand_image_no_2)
        rand_image_no=rand_image_no_2


        self.random_images = [x for x in range(len(self.random_images)) if
                              x not in rand_image_no]  # remove selected images from the choice list

        for img_index in range(len(rand_image_no)):
            deform = 0
            max_dis=0

            imm = self.read_image(rand_image_no[img_index], deform=deform,max_dis=max_dis)
            if len(imm) == 0:

                continue

            self.collection.append(imm)
            print('train image no read so far: %s'%len(self.collection))

        settings.tr_isread=False
        settings.read_patche_mutex_tr.release()



    # --------------------------------------------------------------------------------------------------------
    # read images and transfer those to RAM
    def read_bunch_of_images_vl_from_both_datasets(self, total_sample_no):  # for validation
        if len(settings.bunch_GTV_patches_vl) > total_sample_no:
            return
        if settings.vl_isread == False:
            return
        settings.read_patche_mutex_vl.acquire()
        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)
        self.random_images = list(range(0, len(self.CTs)))
        # select some distinct images for extracting patches!

        rand_image_no1 = np.random.randint(0, 50,
                                           int(self.bunch_of_images_no / 2))  # half from 1st dataset

        rand_image_no2 = np.random.randint(50, len(self.random_images),
                                           int(self.bunch_of_images_no / 2))  # half from 2nd dataset
        rand_image_no = np.hstack((rand_image_no1, rand_image_no2))
        # self.random_images=[x for x in range(len(self.random_images)) if x not in rand_image_no] #remove selected images from the choice list
        print(rand_image_no)
        for img_index in range(len(rand_image_no)):
            if len(settings.bunch_GTV_patches_vl) > total_sample_no:
                self.collection.clear()
                return
            imm = self.read_image(rand_image_no[img_index], deform=0)
            if len(imm) == 0:
                continue

            self.collection.append(imm)
            print('validation read_bunch_of_images: %d' % len(self.collection))
        print('read_bunch_of_images: %d' % len(self.collection))
        settings.vl_isread = False
        settings.read_patche_mutex_vl.release()
    # --------------------------------------------------------------------------------------------------------
    def read_bunch_of_images_vl_from1st_dataset(self, total_sample_no):  # for validation
        if len(settings.bunch_GTV_patches_vl) > total_sample_no:
            return
        if settings.vl_isread == False:
            return
        settings.read_patche_mutex_vl.acquire()
        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)
        self.random_images = list(range(0, len(self.CTs)))
        # select some distinct images for extracting patches!

        rand_image_no1 = np.random.randint(0, len(self.random_images),
                                           int(self.bunch_of_images_no ))  # all from 1st dataset


        rand_image_no = rand_image_no1
        print(rand_image_no)
        for img_index in range(len(rand_image_no)):
            if len(settings.bunch_GTV_patches_vl) > total_sample_no:
                self.collection.clear()
                return
            imm = self.read_image(rand_image_no[img_index], deform=0)
            if len(imm) == 0:
                continue

            self.collection.append(imm)
            print('validation read_bunch_of_images: %d' % len(self.collection))
        print('read_bunch_of_images: %d' % len(self.collection))
        settings.vl_isread = False
        settings.read_patche_mutex_vl.release()
    # ------------------------------------------------------------------------------
    # read images and transfer those to RAM
    def read_bunch_of_images_vl_from2nd_dataset(self, total_sample_no):  # for validation
        if len(settings.bunch_GTV_patches_vl) > total_sample_no:
            return
        if settings.vl_isread == False:
            return
        settings.read_patche_mutex_vl.acquire()
        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)
        self.random_images = list(range(0, len(self.CTs)))
        # select some distinct images for extracting patches!

        rand_image_no1 = np.random.randint(0, len(self.random_images),
                                           int(self.bunch_of_images_no ))  # all from 2nd dataset


        rand_image_no = rand_image_no1
        print(rand_image_no)
        for img_index in range(len(rand_image_no)):
            if len(settings.bunch_GTV_patches_vl) > total_sample_no:
                self.collection.clear()
                return
            imm = self.read_image(rand_image_no[img_index], deform=0)
            if len(imm) == 0:
                continue

            self.collection.append(imm)
            print('validation read_bunch_of_images: %d' % len(self.collection))
        print('read_bunch_of_images: %d' % len(self.collection))
        settings.vl_isread = False
        settings.read_patche_mutex_vl.release()
    # --------------------------------------------------------------------------------------------------------
        # shuffling the patches
    def shuffle_lists(self, CT_image_patchs, GTV_patchs,Penalize_patchs):
            index_shuf = list(range(len(GTV_patchs)))
            shuffle(index_shuf)
            
            CT_image_patchs1 = np.vstack([CT_image_patchs[sn]]
                                         for sn in index_shuf)
            GTV_patchs1 = np.vstack([GTV_patchs[sn]]
                                    for sn in index_shuf)

            Penalize_patchs1 = np.vstack([Penalize_patchs[sn]]
                                             for sn in index_shuf)

            return CT_image_patchs1, GTV_patchs1,Penalize_patchs1
    #--------------------------------------------------------------------------------------------------------
            # read patches from the images which are in the RAM

    def read_patche_online_from_image_bunch_vl(self, sample_no_per_bunch, patch_window, GTV_patchs_size, tumor_percent,
                                            other_percent, img_no):
        if settings.vl_isread == True:
            return

        if len(self.collection) < img_no:
            return
        self.seed += 1
        np.random.seed(self.seed)
        settings.read_patche_mutex_vl.acquire()
        print('start reading:%d' % len(self.collection))
        patch_no_per_image = int(sample_no_per_bunch / len(self.collection))
        # if patch_no_per_image==0:
        #     patch_no_per_image=1
        while patch_no_per_image * len(self.collection) <= sample_no_per_bunch:
            patch_no_per_image += 1
        CT_image_patchs = []
        GTV_patchs = []
        Penalize_patchs = []
        for ii in range(len(self.collection)):
            GTV_image = self.collection[ii].GTV_image
            CT_image = self.collection[ii].CT_image
            Torso_image = self.collection[ii].Torso_image
            Penalize_image = self.collection[ii].Penalize_image

            img_width = self.collection[ii].width
            img_height = self.collection[ii].height
            img_depth = self.collection[ii].depth

            min_torso = self.collection[ii].min_torso
            max_torso = self.collection[ii].max_torso

            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])
            torso_range = np.where(Torso_image == 1)

            '''random numbers for selecting random samples'''
            random_torso = np.random.randint(1, len(torso_range[0]),
                                             size=int(
                                                 patch_no_per_image * other_percent))  # get half depth samples from torso

            rand_depth = torso_range[0][random_torso]
            rand_width = torso_range[1][random_torso]
            rand_height = torso_range[2][random_torso]

            '''balencing the classes:'''
            counter = 0
            rand_depth1 = []  # depth sequence
            rand_width1 = []  # width sequence
            rand_height1 = []  # heigh sequence
            while counter < int(patch_no_per_image * tumor_percent):  # select half of samples from tumor only!
                if (tumor_begin - int(patch_window / 2) - 1 < min_torso):
                    begin = min_torso
                else:
                    begin = tumor_begin

                if (tumor_end + int(patch_window / 2) >= max_torso):
                    end = max_torso
                else:
                    end = tumor_end

                if begin >= end:
                    dpth = [begin]
                else:
                    dpth = np.random.randint(begin, end, size=1)  # select one slice

                if (dpth[0] + int(patch_window / 2) >= GTV_image.shape[0]):
                    dpth = [max_torso]
                if (dpth[0] - int(patch_window / 2) - 1 < 0):
                    dpth = [min_torso]

                ones = np.where(GTV_image[dpth, 0:img_width,
                                0:img_height] != 0)  # GTV indices of slice which belong to tumor

                if len(ones[0]):  # if not empty
                    tmp = int((patch_no_per_image * tumor_percent) / (tumor_end - tumor_begin))
                    if tmp:
                        rnd_ones = np.random.randint(0, len(ones[0]),
                                                     size=tmp)  # number of samples from each slice
                    else:
                        rnd_ones = np.random.randint(0, len(ones[0]),
                                                     size=1)  # number of samples from each slice

                    counter += len(rnd_ones)  # counter for total samples
                    rand_width1 = np.hstack((rand_width1, ones[1][rnd_ones]))
                    rand_depth1 = np.hstack((rand_depth1, dpth * np.ones(len(rnd_ones))))
                    rand_height1 = np.hstack((rand_height1, ones[2][rnd_ones]))

            for sn in range(len(rand_height)):
                if (CT_image[int(rand_depth[sn]) - int(patch_window / 2) - 1:int(rand_depth[sn]) + int(patch_window / 2),
                  int(rand_width[sn]) - int(patch_window / 2) - 1: int(rand_width[sn]) + int(patch_window / 2),
                  int(rand_height[sn]) - int(patch_window / 2) - 1: int(rand_height[sn]) + int(patch_window / 2)]).shape!=(patch_window,patch_window,patch_window):
                    print('problem is shape size')
                    return


            CT_image_patchs1 = np.stack(
                [(CT_image[int(rand_depth[sn]) - int(patch_window / 2) - 1:int(rand_depth[sn]) + int(patch_window / 2),
                  int(rand_width[sn]) - int(patch_window / 2) - 1: int(rand_width[sn]) + int(patch_window / 2),
                  int(rand_height[sn]) - int(patch_window / 2) - 1: int(rand_height[sn]) + int(patch_window / 2)])
                 for sn in range(len(rand_height))])
            GTV_patchs1 = np.stack([(GTV_image[
                                     int(rand_depth[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height[sn]) + int(GTV_patchs_size / 2)
                                     ]).astype(int)
                                    for sn in
                                    range(len(rand_height))]).reshape(len(rand_depth), GTV_patchs_size,
                                                                      GTV_patchs_size, GTV_patchs_size)

            Penalize_patchs1 = np.stack([(Penalize_image[
                                     int(rand_depth[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_height[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height[sn]) + int(GTV_patchs_size / 2)
                                     ]).astype(float)
                                    for sn in
                                    range(len(rand_height))]).reshape(len(rand_depth), GTV_patchs_size,
                                                                      GTV_patchs_size, GTV_patchs_size)
            CT_image_patchs2 = np.stack(
                [(
                 CT_image[int(rand_depth1[sn]) - int(patch_window / 2) - 1:int(rand_depth1[sn]) + int(patch_window / 2),
                 int(rand_width1[sn]) - int(patch_window / 2) - 1: int(rand_width1[sn]) + int(patch_window / 2),
                 int(rand_height1[sn]) - int(patch_window / 2) - 1: int(rand_height1[sn]) + int(patch_window / 2)])
                 for sn in range(len(rand_height1))])
            GTV_patchs2 = np.stack([(GTV_image[
                                     int(rand_depth1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth1[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width1[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                     ]).astype(int)
                                    for sn in
                                    range(len(rand_height1))]).reshape(len(rand_depth1), GTV_patchs_size,
                                                                       GTV_patchs_size, GTV_patchs_size)
            Penalize_patchs2 = np.stack([(Penalize_image[
                                     int(rand_depth1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth1[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width1[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
                                      int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                     ]).astype(float)
                                    for sn in
                                    range(len(rand_height1))]).reshape(len(rand_depth1), GTV_patchs_size,
                                                                       GTV_patchs_size, GTV_patchs_size)
            if len(CT_image_patchs) == 0:
                CT_image_patchs = CT_image_patchs1
                CT_image_patchs = np.vstack((CT_image_patchs, CT_image_patchs2))

                GTV_patchs = GTV_patchs1
                GTV_patchs = np.vstack((GTV_patchs, GTV_patchs2))
                Penalize_patchs=Penalize_patchs1
                Penalize_patchs = np.vstack((Penalize_patchs, Penalize_patchs2))


            else:
                CT_image_patchs = np.vstack((CT_image_patchs, CT_image_patchs1))
                CT_image_patchs = np.vstack((CT_image_patchs, CT_image_patchs2))
                GTV_patchs = np.vstack((GTV_patchs, GTV_patchs1))
                GTV_patchs = np.vstack((GTV_patchs, GTV_patchs2))
                Penalize_patchs = np.vstack((Penalize_patchs, Penalize_patchs1))
                Penalize_patchs = np.vstack((Penalize_patchs, Penalize_patchs2))

            print(len(GTV_patchs))

        if len(GTV_patchs)!=len(Penalize_patchs):
                print(1)
        CT_image_patchs1, GTV_patchs1, Penalize_patchs1 = self.shuffle_lists(CT_image_patchs, GTV_patchs,
                                                                             Penalize_patchs)


        if self.is_training == 1:

            settings.bunch_CT_patches2 = CT_image_patchs1
            settings.bunch_GTV_patches2 = GTV_patchs1
            settings.bunch_Penalize_patches2 = Penalize_patchs1

        else:

            if len(settings.bunch_GTV_patches_vl2) == 0:
                settings.bunch_CT_patches_vl2 = CT_image_patchs1
                settings.bunch_GTV_patches_vl2 = GTV_patchs1
                settings.bunch_Penalize_patches_vl2 = Penalize_patchs1
            else:
                settings.bunch_CT_patches_vl2 = np.vstack((settings.bunch_CT_patches_vl2, CT_image_patchs1))
                settings.bunch_GTV_patches_vl2 = np.vstack((settings.bunch_GTV_patches_vl2, GTV_patchs1))
                settings.bunch_Penalize_patches_vl2 = np.vstack((settings.bunch_Penalize_patches_vl2, Penalize_patchs1))


        settings.vl_isread=True
        settings.read_patche_mutex_vl.release()
        if len(settings.bunch_CT_patches_vl2) != len(
                settings.bunch_GTV_patches_vl2) or len(settings.bunch_Penalize_patches_vl2)!=len(settings.bunch_GTV_patches_vl2 ):
            print('smth wrong')



    #--------------------------------------------------------------------------------------------------------
    #read patches from the images which are in the RAM
    def read_patche_online_from_image_bunch(self, sample_no_per_bunch,patch_window,GTV_patchs_size,tumor_percent,other_percent,img_no):

        if len(self.collection)<img_no:
            return
        if settings.tr_isread == True:
            return
        if len(settings.bunch_GTV_patches)>800:
            return
        self.seed += 1
        np.random.seed(self.seed)
        settings.read_patche_mutex_tr.acquire()
        print('start reading:%d'%len(self.collection))
        patch_no_per_image=int(sample_no_per_bunch/len(self.collection) )
        # if patch_no_per_image==0:
        #     patch_no_per_image=1
        while patch_no_per_image*len(self.collection)<=sample_no_per_bunch:
            patch_no_per_image+=1
        CT_image_patchs=[]
        GTV_patchs=[]
        Penalize_patchs=[]
        for ii in range(len(self.collection) ):
            GTV_image = self.collection[ii].GTV_image
            CT_image = self.collection[ii].CT_image
            Torso_image = self.collection[ii].Torso_image
            Penalize_image = self.collection[ii].Penalize_image

            img_width= self.collection[ii].width
            img_height= self.collection[ii].height

            min_torso = self.collection[ii].min_torso
            max_torso = self.collection[ii].max_torso
            # print(self.collection[ii].ct_name)

            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])
            torso_range=np.where(Torso_image==1)


            '''random numbers for selecting random samples'''
            random_torso=np.random.randint(1,len(torso_range[0]) ,
                                           size=int(patch_no_per_image *other_percent))  # get half depth samples from torso

            rand_depth =torso_range[0][random_torso]
            rand_width =torso_range[1][random_torso]
            rand_height =torso_range[2][random_torso]

            '''balencing the classes:'''
            counter = 0
            rand_depth1 = []  # depth sequence
            rand_width1 = []  # width sequence
            rand_height1 = []  # heigh sequence
            while counter < int(patch_no_per_image * tumor_percent):  # select half of samples from tumor only!
                if (tumor_begin- int(patch_window / 2)-1<min_torso):
                    begin=min_torso
                else:
                    begin = tumor_begin

                if (tumor_end+ int(patch_window / 2)>=max_torso):
                   end= max_torso
                else:
                   end=tumor_end

                if begin>=end:
                    dpth=[begin]
                else:
                    dpth = np.random.randint(begin, end, size=1)  # select one slice

                if (dpth[0]+ int(patch_window / 2)>=GTV_image.shape[0]):
                    dpth = [max_torso]
                if (dpth[0] - int(patch_window / 2) - 1 < 0):
                    dpth = [min_torso]

                ones = np.where(GTV_image[dpth, 0:img_width,
                                0:img_height] != 0)  # GTV indices of slice which belong to tumor

                if len(ones[0]):  # if not empty
                    tmp = int((patch_no_per_image * tumor_percent) / (tumor_end - tumor_begin))
                    if tmp:
                        rnd_ones = np.random.randint(0, len(ones[0]),
                                                     size=tmp)  # number of samples from each slice
                    else:
                        rnd_ones = np.random.randint(0, len(ones[0]),
                                                     size=1)  # number of samples from each slice

                    counter += len(rnd_ones)  # counter for total samples
                    rand_width1 = np.hstack((rand_width1, ones[1][rnd_ones]))
                    rand_depth1 = np.hstack((rand_depth1, dpth * np.ones(len(rnd_ones))))
                    rand_height1 = np.hstack((rand_height1, ones[2][rnd_ones]))


            # for sn in range(len(rand_height)):
            #     tmm=CT_image[int(rand_depth[sn]) - int(patch_window / 2) - 1:int(rand_depth[sn]) + int(patch_window / 2),
            #       int(rand_width[sn]) - int(patch_window / 2) - 1: int(rand_width[sn]) + int(patch_window / 2),
            #       int(rand_height[sn]) - int(patch_window / 2) - 1: int(rand_height[sn]) + int(patch_window / 2)]
            #     if tmm.shape!=(patch_window,patch_window,patch_window):
            #         print(1)



            CT_image_patchs1 = np.stack(
                [(CT_image[int(rand_depth[sn]) - int(patch_window / 2) - 1:int(rand_depth[sn]) + int(patch_window / 2),
                  int(rand_width[sn]) - int(patch_window / 2) - 1: int(rand_width[sn]) + int(patch_window / 2),
                  int(rand_height[sn]) - int(patch_window / 2) - 1: int(rand_height[sn]) + int(patch_window / 2)])
                 for sn in range(len(rand_height))])
            GTV_patchs1 = np.stack([(GTV_image[
                                     int(rand_depth[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height[sn]) + int(GTV_patchs_size / 2)
                                     ] ).astype(int)
                                    for sn in
                                    range(len(rand_height))]).reshape(len(rand_depth), GTV_patchs_size,
                                                            GTV_patchs_size, GTV_patchs_size)
            Penalize_patchs1 = np.stack([(Penalize_image[
                                     int(rand_depth[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height[sn]) + int(GTV_patchs_size / 2)
                                     ] ).astype(float)
                                    for sn in
                                    range(len(rand_height))]).reshape(len(rand_depth), GTV_patchs_size,
                                                            GTV_patchs_size, GTV_patchs_size)

            CT_image_patchs2 = np.stack(
                [(CT_image[int(rand_depth1[sn]) - int(patch_window / 2) - 1:int(rand_depth1[sn]) + int(patch_window / 2),
                  int(rand_width1[sn]) - int(patch_window / 2) - 1: int(rand_width1[sn]) + int(patch_window / 2),
                  int(rand_height1[sn]) - int(patch_window / 2) - 1: int(rand_height1[sn]) + int(patch_window / 2)])
                 for sn in range(len(rand_height1))])
            GTV_patchs2 = np.stack([(GTV_image[
                                     int(rand_depth1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth1[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width1[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                     ] ).astype(int)
                                    for sn in
                                    range(len(rand_height1))]).reshape(len(rand_depth1), GTV_patchs_size,
                                                                      GTV_patchs_size, GTV_patchs_size)

            Penalize_patchs2 = np.stack([(Penalize_image[
                                     int(rand_depth1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth1[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width1[sn]) + int(GTV_patchs_size / 2)
            , int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
              int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                     ]).astype(float)
                                    for sn in
                                    range(len(rand_height1))]).reshape(len(rand_depth1), GTV_patchs_size,
                                                                       GTV_patchs_size, GTV_patchs_size)
            if len(CT_image_patchs)==0:
                CT_image_patchs = CT_image_patchs1
                CT_image_patchs = np.vstack((CT_image_patchs, CT_image_patchs2))

                GTV_patchs = GTV_patchs1
                GTV_patchs = np.vstack((GTV_patchs, GTV_patchs2))

                Penalize_patchs = Penalize_patchs1
                Penalize_patchs = np.vstack((Penalize_patchs, Penalize_patchs2))

            else:
                CT_image_patchs = np.vstack((CT_image_patchs,CT_image_patchs1))
                CT_image_patchs = np.vstack((CT_image_patchs,CT_image_patchs2))
                GTV_patchs = np.vstack((GTV_patchs,GTV_patchs1))
                GTV_patchs = np.vstack((GTV_patchs,GTV_patchs2))
                Penalize_patchs = np.vstack((Penalize_patchs, Penalize_patchs1))
                Penalize_patchs = np.vstack((Penalize_patchs, Penalize_patchs2))

            print(len(GTV_patchs))
            if len(GTV_patchs)!=len(Penalize_patchs):
                print(1)


        CT_image_patchs1, GTV_patchs1,Penalize_patchs1=self.shuffle_lists( CT_image_patchs, GTV_patchs,Penalize_patchs)

        if self.is_training==1:

            settings.bunch_CT_patches2=CT_image_patchs1
            settings.bunch_GTV_patches2=GTV_patchs1
            settings.bunch_Penalize_patches2=Penalize_patchs1

        else:

            if len(settings.bunch_GTV_patches_vl2)==0:
                settings.bunch_CT_patches_vl2=CT_image_patchs1
                settings.bunch_GTV_patches_vl2=GTV_patchs1
                settings.bunch_Penalize_patches_vl2=Penalize_patchs1
            else:
                settings.bunch_CT_patches_vl2 = np.vstack((settings.bunch_CT_patches_vl2,CT_image_patchs1))
                settings.bunch_GTV_patches_vl2 = np.vstack((settings.bunch_GTV_patches_vl2,GTV_patchs1))
                settings.bunch_Penalize_patches_vl2 = np.vstack((settings.bunch_Penalize_patches_vl2,Penalize_patchs1))


        settings.tr_isread=True
        settings.read_patche_mutex_tr.release()
        if len(settings.bunch_CT_patches_vl2)!=len(settings.bunch_GTV_patches_vl2) or len(settings.bunch_Penalize_patches_vl2)!=len(settings.bunch_GTV_patches_vl2 ):
            print('smth wrong')

    #--------------------------------------------------------------------------------------------------------
    def read_patche_online_from_image_bunch_with_trackno(self, sample_no_per_bunch,patch_window,GTV_patchs_size,tumor_percent,other_percent,img_no):

        if len(self.collection)<img_no:
            return
        if settings.tr_isread == True:
            return
        if len(settings.bunch_GTV_patches)>800:
            return
        self.seed += 1
        np.random.seed(self.seed)
        settings.read_patche_mutex_tr.acquire()
        print('start reading:%d'%len(self.collection))
        patch_no_per_image=int(sample_no_per_bunch/len(self.collection) )
        # if patch_no_per_image==0:
        #     patch_no_per_image=1
        while patch_no_per_image*len(self.collection)<=sample_no_per_bunch:
            patch_no_per_image+=1
        CT_image_patchs=[]
        GTV_patchs=[]
        Penalize_patchs=[]
        for ii in range(len(self.collection) ):
            GTV_image = self.collection[ii].GTV_image
            CT_image = self.collection[ii].CT_image
            Torso_image = self.collection[ii].Torso_image
            Penalize_image = self.collection[ii].Penalize_image

            img_width= self.collection[ii].width
            img_height= self.collection[ii].height

            min_torso = self.collection[ii].min_torso
            max_torso = self.collection[ii].max_torso
            # print(self.collection[ii].ct_name)

            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])
            torso_range=np.where(Torso_image==1)


            '''random numbers for selecting random samples'''
            random_torso=np.random.randint(1,len(torso_range[0]) ,
                                           size=int(patch_no_per_image *other_percent))  # get half depth samples from torso

            rand_depth =torso_range[0][random_torso]
            rand_width =torso_range[1][random_torso]
            rand_height =torso_range[2][random_torso]

            '''balencing the classes:'''
            counter = 0
            rand_depth1 = []  # depth sequence
            rand_width1 = []  # width sequence
            rand_height1 = []  # heigh sequence
            while counter < int(patch_no_per_image * tumor_percent):  # select half of samples from tumor only!
                if (tumor_begin- int(patch_window / 2)-1<min_torso):
                    begin=min_torso
                else:
                    begin = tumor_begin

                if (tumor_end+ int(patch_window / 2)>=max_torso):
                   end= max_torso
                else:
                   end=tumor_end

                if begin>=end:
                    dpth=[begin]
                else:
                    dpth = np.random.randint(begin, end, size=1)  # select one slice

                if (dpth[0]+ int(patch_window / 2)>=GTV_image.shape[0]):
                    dpth = [max_torso]
                if (dpth[0] - int(patch_window / 2) - 1 < 0):
                    dpth = [min_torso]

                ones = np.where(GTV_image[dpth, 0:img_width,
                                0:img_height] != 0)  # GTV indices of slice which belong to tumor

                if len(ones[0]):  # if not empty
                    tmp = int((patch_no_per_image * tumor_percent) / (tumor_end - tumor_begin))
                    if tmp:
                        rnd_ones = np.random.randint(0, len(ones[0]),
                                                     size=tmp)  # number of samples from each slice
                    else:
                        rnd_ones = np.random.randint(0, len(ones[0]),
                                                     size=1)  # number of samples from each slice

                    counter += len(rnd_ones)  # counter for total samples
                    rand_width1 = np.hstack((rand_width1, ones[1][rnd_ones]))
                    rand_depth1 = np.hstack((rand_depth1, dpth * np.ones(len(rnd_ones))))
                    rand_height1 = np.hstack((rand_height1, ones[2][rnd_ones]))


            # for sn in range(len(rand_height)):
            #     tmm=CT_image[int(rand_depth[sn]) - int(patch_window / 2) - 1:int(rand_depth[sn]) + int(patch_window / 2),
            #       int(rand_width[sn]) - int(patch_window / 2) - 1: int(rand_width[sn]) + int(patch_window / 2),
            #       int(rand_height[sn]) - int(patch_window / 2) - 1: int(rand_height[sn]) + int(patch_window / 2)]
            #     if tmm.shape!=(patch_window,patch_window,patch_window):
            #         print(1)



            CT_image_patchs1 = np.stack(
                [(CT_image[int(rand_depth[sn]) - int(patch_window / 2) - 1:int(rand_depth[sn]) + int(patch_window / 2),
                  int(rand_width[sn]) - int(patch_window / 2) - 1: int(rand_width[sn]) + int(patch_window / 2),
                  int(rand_height[sn]) - int(patch_window / 2) - 1: int(rand_height[sn]) + int(patch_window / 2)])
                 for sn in range(len(rand_height))])
            GTV_patchs1 = np.stack([(GTV_image[
                                     int(rand_depth[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height[sn]) + int(GTV_patchs_size / 2)
                                     ] ).astype(int)
                                    for sn in
                                    range(len(rand_height))]).reshape(len(rand_depth), GTV_patchs_size,
                                                            GTV_patchs_size, GTV_patchs_size)
            Penalize_patchs1 = np.stack([(Penalize_image[
                                     int(rand_depth[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height[sn]) + int(GTV_patchs_size / 2)
                                     ] ).astype(float)
                                    for sn in
                                    range(len(rand_height))]).reshape(len(rand_depth), GTV_patchs_size,
                                                            GTV_patchs_size, GTV_patchs_size)

            CT_image_patchs2 = np.stack(
                [(CT_image[int(rand_depth1[sn]) - int(patch_window / 2) - 1:int(rand_depth1[sn]) + int(patch_window / 2),
                  int(rand_width1[sn]) - int(patch_window / 2) - 1: int(rand_width1[sn]) + int(patch_window / 2),
                  int(rand_height1[sn]) - int(patch_window / 2) - 1: int(rand_height1[sn]) + int(patch_window / 2)])
                 for sn in range(len(rand_height1))])
            GTV_patchs2 = np.stack([(GTV_image[
                                     int(rand_depth1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth1[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width1[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                     ] ).astype(int)
                                    for sn in
                                    range(len(rand_height1))]).reshape(len(rand_depth1), GTV_patchs_size,
                                                                      GTV_patchs_size, GTV_patchs_size)

            Penalize_patchs2 = np.stack([(Penalize_image[
                                     int(rand_depth1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_depth1[sn]) + int(GTV_patchs_size / 2),
                                     int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width1[sn]) + int(GTV_patchs_size / 2)
            , int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
              int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                     ]).astype(float)
                                    for sn in
                                    range(len(rand_height1))]).reshape(len(rand_depth1), GTV_patchs_size,
                                                                       GTV_patchs_size, GTV_patchs_size)
            if len(CT_image_patchs)==0:
                CT_image_patchs = CT_image_patchs1
                CT_image_patchs = np.vstack((CT_image_patchs, CT_image_patchs2))

                GTV_patchs = GTV_patchs1
                GTV_patchs = np.vstack((GTV_patchs, GTV_patchs2))

                Penalize_patchs = Penalize_patchs1
                Penalize_patchs = np.vstack((Penalize_patchs, Penalize_patchs2))

            else:
                CT_image_patchs = np.vstack((CT_image_patchs,CT_image_patchs1))
                CT_image_patchs = np.vstack((CT_image_patchs,CT_image_patchs2))
                GTV_patchs = np.vstack((GTV_patchs,GTV_patchs1))
                GTV_patchs = np.vstack((GTV_patchs,GTV_patchs2))
                Penalize_patchs = np.vstack((Penalize_patchs, Penalize_patchs1))
                Penalize_patchs = np.vstack((Penalize_patchs, Penalize_patchs2))

            print(len(GTV_patchs))
            if len(GTV_patchs)!=len(Penalize_patchs):
                print(1)


        CT_image_patchs1, GTV_patchs1,Penalize_patchs1=self.shuffle_lists( CT_image_patchs, GTV_patchs,Penalize_patchs)

        if self.is_training==1:

            settings.bunch_CT_patches2=CT_image_patchs1
            settings.bunch_GTV_patches2=GTV_patchs1
            settings.bunch_Penalize_patches2=Penalize_patchs1

        else:

            if len(settings.bunch_GTV_patches_vl2)==0:
                settings.bunch_CT_patches_vl2=CT_image_patchs1
                settings.bunch_GTV_patches_vl2=GTV_patchs1
                settings.bunch_Penalize_patches_vl2=Penalize_patchs1
            else:
                settings.bunch_CT_patches_vl2 = np.vstack((settings.bunch_CT_patches_vl2,CT_image_patchs1))
                settings.bunch_GTV_patches_vl2 = np.vstack((settings.bunch_GTV_patches_vl2,GTV_patchs1))
                settings.bunch_Penalize_patches_vl2 = np.vstack((settings.bunch_Penalize_patches_vl2,Penalize_patchs1))


        settings.tr_isread=True
        settings.read_patche_mutex_tr.release()
        if len(settings.bunch_CT_patches_vl2)!=len(settings.bunch_GTV_patches_vl2) or len(settings.bunch_Penalize_patches_vl2)!=len(settings.bunch_GTV_patches_vl2 ):
            print('smth wrong')
    #--------------------------------------------------------------------------------------------------------
    def return_patches(self,batch_no):
        settings.train_queue.acquire()
        CT_patch=[]
        GTv_patch=[]
        loss_coef=[]
        Penalize_patch=[]
        if len(settings.bunch_CT_patches)>=batch_no and\
            len(settings.bunch_GTV_patches) >= batch_no  :
            # \                        len(settings.bunch_Penalize_patches) >= batch_no:
            CT_patch=settings.bunch_CT_patches[0:batch_no]
            GTv_patch=settings.bunch_GTV_patches[0:batch_no]
            Penalize_patch=settings.bunch_Penalize_patches[0:batch_no]

            settings.bunch_CT_patches=np.delete(settings.bunch_CT_patches,range(batch_no),axis=0)
            settings.bunch_GTV_patches=np.delete(settings.bunch_GTV_patches,range(batch_no),axis=0)
            settings.bunch_Penalize_patches=np.delete(settings.bunch_Penalize_patches,range(batch_no),axis=0)
            GTv_patch = np.eye(2)[GTv_patch]
            CT_patch = CT_patch[..., np.newaxis]
            Penalize_patch = Penalize_patch[..., np.newaxis]
            loss_coef = ( np.asarray([[len(np.where(GTv_patch[i, :, :, :] == 1)[0]) / np.power(self.gtv_patch_window, 3)] for i in range(GTv_patch.shape[0])]))
            loss_coef = np.hstack((loss_coef, 1 - loss_coef)) #tumor vs non-tumor

        else:
            settings.bunch_CT_patches = np.delete(settings.bunch_CT_patches, range(len(settings.bunch_CT_patches)), axis=0)
            settings.bunch_GTV_patches = np.delete(settings.bunch_GTV_patches, range(len(settings.bunch_GTV_patches)), axis=0)
            settings.bunch_Penalize_patches = np.delete(settings.bunch_Penalize_patches, range(len(settings.bunch_Penalize_patches)), axis=0)
        settings.train_queue.release()
        if len(CT_patch)!=len(GTv_patch) |len(CT_patch)!=len(Penalize_patch):
            print('smth wrong')


        return CT_patch,GTv_patch,Penalize_patch,loss_coef



    #--------------------------------------------------------------------------------------------------------
    def return_patches_validation(self, start,end):
            CT_patch = []
            GTv_patch = []
            Penalize_patch = []

            if (len(settings.bunch_CT_patches_vl)-(end)) >= 0\
                    and (len(settings.bunch_GTV_patches_vl)-(end)) >= 0 \
                    and (len(settings.bunch_Penalize_patches_vl) - (end)) >= 0:
                CT_patch = settings.bunch_CT_patches_vl[start:end]
                GTv_patch = settings.bunch_GTV_patches_vl[start:end]
                Penalize_patch = settings.bunch_Penalize_patches_vl[start:end]

                if len(CT_patch) != len(GTv_patch) or len(Penalize_patch) != len(CT_patch):
                    print('smth wrong')


                # loss_coef = ( np.asarray([[len(np.where(GTv_patch[i, :, :, :] == 1)[0]) / np.power(self.gtv_patch_window, 3)] for i in range(GTv_patch.shape[0])]))

                GTv_patch = np.eye(2)[GTv_patch]
                CT_patch = CT_patch[..., np.newaxis]
                Penalize_patch = Penalize_patch[..., np.newaxis]

            return CT_patch, GTv_patch,Penalize_patch
    # -------------------------------------------------------------------------------------------------------
