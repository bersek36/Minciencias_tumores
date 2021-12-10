import os

import numpy as np
from numpy import savez_compressed


def get_training_sub_volumes(image, image_affine, label, label_affine, 
                   subvol_dest_folder, submask_dest_folder, rotation_matrix_path, classes=1,
                   orig_x = 240, orig_y = 240, orig_z = 48, 
                   output_x = 80, output_y = 80, output_z = 16,
                   stride_x = 40, stride_y = 40, stride_z = 8,
                   background_threshold=0.1):

    X = None
    y = None
    
    tries = 0
    savez_compressed(rotation_matrix_path, image_affine.astype(np.float32))
    for i_2 in range(0, orig_z-output_z+2, stride_z):#(0, orig_x-output_x+1, output_x-1):
        if i_2 < 124:
            i = i_2
        else:
            #print(orig_z-output_z+2)
            i = i_2-1
        for j in range(0, orig_y-output_y+1, stride_y):
            for k in range(0, orig_x-output_x+1, stride_x):

                # extract relevant area of label
                y = label[k: k + output_x,
                          j: j + output_y,
                          i: i + output_z]
                
                volume = output_x*output_y*output_z

                # compute the background ratio 
                bgrd_ratio = np.sum(y==classes)/volume

                # if background ratio is above the threshold, use that sub-volume. otherwise continue and try another sub-volume
                if bgrd_ratio >= background_threshold: # Lo que se quiere segmentar se al menos threshold%
                    #print(bgrd_ratio)
                    tries += 1
                    X = image[k: k + output_x,
                          j: j + output_y,
                          i: i + output_z]
                    savez_compressed(os.path.join(subvol_dest_folder, 'subvolume'+str(tries)), X.astype(np.float32))
                    #nii_subvolume = nib.Nifti1Image(X.astype(np.float32), image_affine)
                    #nib.save(nii_subvolume, os.path.join(subvol_dest_folder, 'subvolume'+str(tries)+'.nii'))
                    savez_compressed(os.path.join(submask_dest_folder, 'submask'+str(tries)), y.astype(np.float32))
                    #nii_submask = nib.Nifti1Image(y.astype(np.float32), label_affine)
                    #nib.save(nii_submask, os.path.join(submask_dest_folder, 'submask'+str(tries)+'.nii'))

def get_test_subvolumes(image, image_affine, label, label_affine, 
                        subvol_dest_folder, submask_dest_folder,rotation_matrix_path, 
                        orig_x = 240, orig_y = 240, orig_z = 48, 
                        output_x = 80, output_y = 80, output_z = 16,
                        stride_x = 40, stride_y = 40, stride_z = 8):
    
    X = None
    y = None
    
    tries = 0
    savez_compressed(rotation_matrix_path, image_affine.astype(np.float32))
    for i_2 in range(0, orig_z-output_z+2, stride_z):#(0, orig_x-output_x+1, output_x-1):
        if i_2 < 124:
            i = i_2
        else:
            #print(orig_z-output_z+2)
            i = i_2-1
        for j in range(0, orig_y-output_y+1, stride_y):
            for k in range(0, orig_x-output_x+1, stride_x):
                tries += 1

                # extract relevant area of label
                y = label[k: k + output_x,
                          j: j + output_y,
                          i: i + output_z]

                X = image[k: k + output_x,
                      j: j + output_y,
                      i: i + output_z]
                
                savez_compressed(os.path.join(subvol_dest_folder, 'subvolume'+str(tries)), X.astype(np.float32))
                savez_compressed(os.path.join(submask_dest_folder, 'submask'+str(tries)), y.astype(np.float32))
                #nii_subvolume = nib.Nifti1Image(X.astype(np.float32), image_affine)
                #nib.save(nii_subvolume, os.path.join(subvol_dest_folder, 'subvolume'+str(tries)+'.nii'))
                #nii_submask = nib.Nifti1Image(y.astype(np.float32), label_affine)
                #nib.save(nii_submask, os.path.join(submask_dest_folder, 'submask'+str(tries)+'.nii'))

                  
