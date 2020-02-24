from densenet_classify2 import dense_classify

#using nothing

dc12=dense_classify( data=2,densnet_unet_config=[3,7,9,7,3], compression_coefficient=1,growth_rate=4,
                 sample_no=2000000,validation_samples=4000,no_sample_per_each_itr=2000,
                 train_tag='train/', validation_tag='validation/', test_tag='test/',
                           img_name='CT_padded.mha',label_name='GTV_CT_padded.mha', torso_tag='CT_padded_Torso.mha',
                     log_tag='_DenseUnet_esoph_lessFM_11',min_range=-1000,max_range=3000,
                     tumor_percent=.75,other_percent=.25)
dc12.run_net()


# # using nothing
# dc12=dense_classify( data=1,densnet_unet_config=[3,4,4,4,3], compression_coefficient=.5,growth_rate=4,
#                  sample_no=2280000,validation_samples=5700,no_sample_per_each_itr=3420,
#                  train_tag='prostate_train/', validation_tag='prostate_validation/', test_tag='prostate_test/',
#                            img_name='CTImage.mha',label_name='Bladder.mha', torso_tag='Torso.mha',log_tag='_dense_net_prostate_3D',min_range=-1000,max_range=3000)
# dc12.run_net()
#
# # using resampling:
# dc12=dense_classify( data=1,densnet_unet_config=[3,4,4,4,3], compression_coefficient=.5,growth_rate=4,
#                  sample_no=2280000,validation_samples=5700,no_sample_per_each_itr=3420,
#                  train_tag='prostate_train/', validation_tag='prostate_validation/', test_tag='prostate_test/',
#                            img_name='CTImage_re222.mha',label_name='Bladder_re222.mha', torso_tag='Torso.mha',log_tag='_dense_net_prostate_3D_re222',min_range=-1000,max_range=3000)
# dc12.run_net()


