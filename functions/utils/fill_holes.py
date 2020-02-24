#!/usr/bin/env python
# import itk
import SimpleITK as itk

# if len(sys.argv) != 3:
#     print("Usage: " + sys.argv[0] + " <InputFileName> <OutputFileName>")
#     sys.exit(1)
path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/dense_net_14/densenet_unet_output_volume_68886/'
inputFileName=path+'Patient_22_visit_20071026.mha'
outputFileName=path+'pp.mha'


Dimension = 2

PixelType = itk.UC
ImageType = itk.Image[PixelType, Dimension]

ReaderType = itk.ImageFileReader[ImageType]
reader = ReaderType.New()
reader.SetFileName(inputFileName)

FilterType = itk.BinaryFillholeImageFilter[ImageType]
binaryfillholefilter = FilterType.New()
binaryfillholefilter.SetInput(reader.GetOutput())
binaryfillholefilter.SetForegroundValue(itk.NumericTraits[PixelType].min())

WriterType = itk.ImageFileWriter[ImageType]
writer = WriterType.New()
writer.SetFileName(outputFileName)
writer.SetInput(binaryfillholefilter.GetOutput())
writer.Update()