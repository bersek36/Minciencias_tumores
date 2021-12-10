import numpy as np
import nibabel as nib
import itertools
import os
from multiprocessing import Pool
from timeit import default_timer as timer
def resize_data(data):
    initial_size_x = data.shape[0]
    initial_size_y = data.shape[1]
    initial_size_z = data.shape[2]

    new_size_x = 256
    new_size_y = 256
    new_size_z = 192

    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y
    delta_z = initial_size_z / new_size_z

    new_data = np.zeros((new_size_x, new_size_y, new_size_z))

    for x, y, z in itertools.product(range(new_size_x),
                                     range(new_size_y),
                                     range(new_size_z)):
        new_data[x][y][z] = data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

    return new_data.astype(np.int16)


#os.chdir("/home/mri/3T_extracted_ad/")
#ad_files = os.listdir()

#for file in ad_files:
def resize(i):
    num = "000"
    if i+1 < 10:
        num = "00{}".format(i+1)
    elif i+1 < 100:
        num = "0{}".format(i+1)
    else:
        num = "{}".format(i+1)
    t1 = "E:\BRATS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_{}\BraTS20_Training_{}_t1.nii".format(num, num)
    msk = "E:\BRATS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_{}\BraTS20_Training_{}_seg.nii".format(num, num)
    out_file = "E:\BRATS\BraTS2020_TrainingData\\resized\BraTS20_Training_{}".format(num)
    os.makedirs(out_file)
    initial_t1 = nib.load(t1).get_fdata()
    initial_msk = nib.load(msk).get_fdata()
    #if initial_data.shape != (256, 256, 192):
    resized_t1 = resize_data(initial_t1)
    resized_msk = resize_data(initial_msk)
    img_t1 = nib.Nifti1Image(resized_t1, np.eye(4))
    img_msk = nib.Nifti1Image(resized_msk, np.eye(4))
    img_t1.to_filename(out_file + "\BraTS20_Training_{}_t1.nii".format(num))
    img_msk.to_filename(out_file + "\BraTS20_Training_{}_seg.nii".format(num))


if __name__ == '__main__':
    numeros =  list(range(0, 369))
    start = timer()
    with Pool(6) as pool:
        print("start")
        pool.map(resize,numeros)
        print("finished")
    end = timer()
    
    print("Elapsed time {}".format(end-start))