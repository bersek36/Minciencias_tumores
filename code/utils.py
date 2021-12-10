import os
from multiprocessing import Pool
from timeit import default_timer as timer

import nibabel as nib
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import load, savez_compressed
from natsort import natsorted

from preprocess.get_subvolume import get_training_sub_volumes, get_test_subvolumes
# TODO: Condicionar la creacion de los directorios para separar train y test en carpetas diferentes
back_threshold = 0.0
random_state = 42
orig_x = 256
orig_y = 256
orig_z = 192 

output_x = 64
output_y = 64
output_z = 16

stride_x = 64
stride_y = 64
stride_z = 16
def make_dirs():
    """Construye los directorios donde se almacenan todos los datos en caso de que no existan aun.
    Como resultado retorna un diccionario con los paths.

    claves del diccionario: 
    ['SUBVOLUME_FOLDER', 'SUBVOLUME_MASK_FOLDER', 'RESULTADOS', 'DATABASE_DIR', 'SAMPLES']

    :return: diccionario con los paths 
    :rtype: dict
    """
    processed_paths = {}
    processed_sample_paths = {}
    PARENT_DIR = os.getcwd()
    DATABASE_DIR = os.path.join(PARENT_DIR, "NFBS_Dataset")
    SAMPLES_FOLDERS = next(os.walk(DATABASE_DIR))[1]
    
    processed_paths["PROCESSED_DIR"] = os.path.join(PARENT_DIR, "processed")
    PROCESSED_DIR = processed_paths["PROCESSED_DIR"]

    processed_paths["3D"] = os.path.join(PROCESSED_DIR, "3D")
    UNET_3D = processed_paths["3D"]

    processed_paths["PRUEBA"] = os.path.join(PROCESSED_DIR, "prueba")
    UNET_PRUEBA = processed_paths["PRUEBA"]

    processed_paths["2D"] = os.path.join(PROCESSED_DIR, "2D")
    UNET_2D = processed_paths["2D"]
    processed_paths["SUBVOLUME_FOLDER"] = os.path.join(UNET_3D,"subvolumes")
    processed_paths["SUBVOLUME_MASK_FOLDER"] = os.path.join(UNET_3D,"subvolumes_masks")
    processed_paths["ROTATION_MATRIX"] = os.path.join(UNET_3D,"rotation_matrix")
    processed_paths["RESULTS_3D"] = os.path.join(UNET_3D,"subvolumes_predict")

    processed_paths["IMAGES_FOLDER"] = os.path.join(UNET_2D,"images")
    processed_paths["IMAGES_MASK_FOLDER"] = os.path.join(UNET_2D,"images_masks")
    processed_paths["ROTATION_MATRIX_2D"] = os.path.join(UNET_2D,"rotation_matrix")
    processed_paths["RESULTS_2D"] = os.path.join(UNET_2D,"images_predict")

    processed_paths["IMAGES_FOLDER_PRUEBA"] = os.path.join(UNET_PRUEBA,"images")
    processed_paths["IMAGES_MASK_FOLDER_PRUEBA"] = os.path.join(UNET_PRUEBA,"images_masks")
    processed_paths["ROTATION_MATRIX_2D_PRUEBA"] = os.path.join(UNET_PRUEBA,"rotation_matrix")
    processed_paths["RESULTS_2D_PRUEBA"] = os.path.join(UNET_PRUEBA,"images_predict")
    
    for path in processed_paths:
        dir_exists = os.path.exists(processed_paths[path])
        if not dir_exists:
            os.makedirs(processed_paths[path])

    processed_paths.pop("PROCESSED_DIR")
    processed_paths.pop("3D")
    processed_paths.pop("2D")
    processed_paths.pop("PRUEBA")
    for path in processed_paths:
        if path == "RESULTS_3D" or path == "RESULTS_2D":
            train_files, val_files = train_test_split(SAMPLES_FOLDERS, test_size=0.2, random_state=42)
            test_files, val_files = train_test_split(val_files, test_size=0.5, random_state=42)
            SAMPLES_FOLDERS_TEST_TRAIN = test_files
            TEST_SAMPLES = SAMPLES_FOLDERS_TEST_TRAIN
        else:
            SAMPLES_FOLDERS_TEST_TRAIN = SAMPLES_FOLDERS

        for sample in SAMPLES_FOLDERS_TEST_TRAIN:
            sample_dir = os.path.join(processed_paths[path], sample)
            dir_exists = os.path.exists(sample_dir)
            if not dir_exists:
                os.makedirs(sample_dir)
        processed_sample_paths[path] = processed_paths[path]
    processed_sample_paths["DATABASE_DIR"] = DATABASE_DIR
    processed_sample_paths["SAMPLES"] = SAMPLES_FOLDERS
    processed_sample_paths["TEST_SAMPLES"] = TEST_SAMPLES
    return processed_sample_paths, train_files, val_files, test_files

# Se crean los directorios
# Estos paths se usan en las funciones
paths, train_files, val_files, test_files = make_dirs()
test_files.extend(val_files)

def create_file_name(sample, mask=1):
    """Funcion que genera el nombre de los archivos para automatizar la lectura

    :param sample: codigo del paciente (carpeta)
    :type sample: string
    :param mask: Define si se quiere la mascara o el volumen normal, defaults to False
    :type mask: bool, optional
    :return: nombre del archivo a leer
    :rtype: path
    """
    if mask==1:
        file_name = "sub-"+sample+"_ses-NFB3_T1w_brainmask.nii.gz"
    elif mask==2:
        file_name = "sub-"+sample+"_ses-NFB3_T1w.nii.gz"
    
    elif mask==3:
        file_name = "sub-"+sample+"_ses-NFB3_T1w_brain.nii.gz"
    file_name = os.path.join(sample, file_name)
    return file_name

def get_sub_volumes_train(sample):
    """Esta funcion recibe el nombre de una muestra (carpeta) y genera los sub-volumenes.
    SOLO PARA TRAIN!!

    :param sample: nombre de la carpeta que contiene el volumen y su mascara
    :type sample: path
    """
    img = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, 1))
    img_mask = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, 2))
    img = nib.load(img)
    img_mask = nib.load(img_mask)
    image = img.get_fdata()
    image_mask = img_mask.get_fdata()
    SAVE_PATH_SUBVOLUME = os.path.join(paths["SUBVOLUME_FOLDER"], sample)
    SAVE_PATH_SUBMASK = os.path.join(paths["SUBVOLUME_MASK_FOLDER"], sample)
    ROTATION_MATRIX_PATH = os.path.join(paths["ROTATION_MATRIX"], sample,sample)
    get_training_sub_volumes(image, img.affine, image_mask, img_mask.affine, 
                                    SAVE_PATH_SUBVOLUME, SAVE_PATH_SUBMASK, ROTATION_MATRIX_PATH,
                                    classes=1, 
                                    orig_x = 256, orig_y = 256, orig_z = 192, 
                                    output_x = output_x, output_y = output_y, output_z = output_z,
                                    stride_x = stride_x, stride_y = stride_y, stride_z = stride_z,
                                    background_threshold=back_threshold)

def get_sub_volumes_test(sample):
    """Esta funcion recibe el nombre de una muestra (carpeta) y genera los sub-volumenes.
    SOLO PARA TEST!!

    NOTA: el stride de los volumenes de test es mas grande 
    debido a que genera muchas mas imagenes y no tengo espacio.

    :param sample: nombre de la carpeta que contiene el volumen y su mascara
    :type sample: path
    """
    img = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, 1))
    img_mask = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, 2))
    img = nib.load(img)
    img_mask = nib.load(img_mask)
    image = img.get_fdata()
    image_mask = img_mask.get_fdata()
    SAVE_PATH_SUBVOLUME = os.path.join(paths["SUBVOLUME_FOLDER"], sample)
    SAVE_PATH_SUBMASK = os.path.join(paths["SUBVOLUME_MASK_FOLDER"], sample)
    ROTATION_MATRIX_PATH = os.path.join(paths["ROTATION_MATRIX"], sample,sample)
    get_test_subvolumes(image, img.affine, image_mask, img_mask.affine, 
                                    SAVE_PATH_SUBVOLUME, SAVE_PATH_SUBMASK, ROTATION_MATRIX_PATH,
                                    orig_x = 256, orig_y = 256, orig_z = 192, 
                                    output_x = output_x, output_y = output_y, output_z = output_z,
                                    stride_x = stride_x, stride_y = stride_y, stride_z = stride_z)


def reconstruction(orig_x = 256, orig_y = 256,orig_z = 192,
                   output_x = output_x, output_y = output_y, output_z = output_z,
                   stride_x = stride_x, stride_y = stride_y, stride_z = stride_z,
                   test_files=None, path_resultados=None, path_rotation_matrix=None, Unet_2D=False):
    """Esta funcion toma una lista con los nombres de las carpetas que contienen los 
    sub-volumenes de cada paciente y el path donde se almacenan los sub-volumenes de test.
    Luego realiza la reconstruccion de los volumenes a su tamaño original.

    Guarda las reconstrucciones en path_resultados con el nombre de cada muestra (el mismo de la carpeta)

    :param orig_x: tamaño original del volumen en X, defaults to 256
    :type orig_x: int, optional
    :param orig_y: tamaño original del volumen en Y, defaults to 256
    :type orig_y: int, optional
    :param orig_z: tamaño original del volumen en Z, defaults to 192
    :type orig_z: int, optional
    :param output_x: tamaño final de cada sub-volumen en X, defaults to 128
    :type output_x: int, optional
    :param output_y: tamaño final de cada sub-volumen en Y, defaults to 128
    :type output_y: int, optional
    :param output_z: tamaño final de cada sub-volumen en Z, defaults to 16
    :type output_z: int, optional
    :param stride_x: Stride en el eje X, defaults to 128
    :type stride_x: int, optional
    :param stride_y: Stride en el eje Y, defaults to 128
    :type stride_y: int, optional
    :param stride_z: Stride en el eje Z, defaults to 8
    :type stride_z: int, optional
    :param test_files: Contiene los nombres de las carpetas con los volumenes de cada paciente, defaults to None
    :type test_files: python list, optional
    :param path_resultados: Contiene el path donde estan las mascaras predichas para los sub-volumenes de test, defaults to None
    :type path_resultados: string, optional

    Ejemplo de uso:

    test_files = ['A00057965','A00055373','A00055542']
    paths["RESULTADOS"] = "c:\\Users\\Bersek\\Desktop\\proyecto_minciencias\\Minciencias_pruebas\\processed\\subvolumes_predicts"
    paths["ROTATION_MATRIX"] = "c:\\Users\\Bersek\\Desktop\\proyecto_minciencias\\Minciencias_pruebas\\processed\\rotation_matrix"

    reconstruction(test_files=test_files, path_resultados=paths["RESULTS"],path_rotation_matrix=paths["ROTATION_MATRIX"] , stride_z=8)

    """
    z = 0
    for i in range(0, orig_z-output_z+1, stride_z):
        z+=1
        y = 0
        for j in range(0, orig_y-output_y+1, stride_y):
            y+=1
            x = 0
            for k in range(0, orig_x-output_x+1, stride_x):
                x+=1
    print("X:",x, " Y:",y, " Z:",z)
    for sample in test_files:
        path = []
        flag = False
        num = 0
        tries = 0

        for subvol in os.listdir(os.path.join(path_resultados,sample)):
            path.append(os.path.join(path_resultados,sample,subvol))
        path = natsorted(path)
        
        for i in range(0, z):
            for j in range(0, y):
                for k in range(0, x):
                    dict_data = load(path[tries])
                    
                    # extract the first array
                    nifti = dict_data['arr_0']
                    tries += 1 # Esta variable se usa para el indice de los sub-columenes
                    ###########################################################################################
                    ## CAJA NEGRA ##
                    """
                    Explicación de la caja negra:
                    Imaginemos el volumen como un cubo 3D (eso es lo que realmente es). 
                    Tomamos la primera cara del volumen en la primera posición. (num=0)
                    1*******
                    ********
                    ********
                    ********
                    Si estamos ahí se guarda ese parche en <array_aux>, luego nos movemos a lo largo del eje X. (num=1)
                    *2******
                    ********
                    ********
                    ********
                    Concatenamos <array_aux> con la nueva imagen teniendo en cuenta el stride y guardamos en <array_aux> nuevamente.

                    Cuando ya completamos todo el eje X nos movemos en el eje Y y repetimos el proceso en el eje X.(num=2)
                    Si es la primera vez se concatenan <array_aux2>, <array_aux> y se guarda en <array_nuevo2> teniendo en cuenta el stride
                    ********
                    2*******
                    ********
                    ********
                    Luego se concatenan <array_nuevo2> y <array_aux> teniendo en cuenta el stride
                    ********
                    ********
                    3*******
                    ********
                    En el caso del eje Z se sigue la misma logica solo que se usa una bandera booleana. (num=3)
                    """
                    if num == 0:
                        array_aux = nifti 
                        num = 1
                    elif num == 1:
                        array_aux = np.concatenate((array_aux, nifti[output_x-stride_x:output_x,:,:]), axis=0)
                        if k >= x-1:
                            if j>0:
                                num = 2
                            else:
                                array_aux2 = array_aux
                                num = 0
                            
                    if num == 2:
                        if j == 1:
                            array_nuevo2 = np.concatenate((array_aux2, array_aux[:,output_y-stride_y:output_y,:]), axis=1)
                            
                        else:
                            array_nuevo2 = np.concatenate((array_nuevo2, array_aux[:,output_y-stride_y:output_y,:]), axis=1)
                        num = 0

                        if j >= y-1:
                            num = 3

                    if num==3 and not flag:
                        array_nuevo3 = array_nuevo2
                        flag=True
                        num = 0

                    elif num==3 and  flag:
                        array_nuevo3 = np.concatenate((array_nuevo3, array_nuevo2[:,:,output_z-stride_z:output_z]), axis=2)
                        num = 0
                    ##FINAL DE LA CAJA NEGRA ##
                    ###########################################################################################
        # Se guarda la info
        dict_data = load(os.path.join(path_rotation_matrix,sample,sample+".npz"))
        img_affine = dict_data["arr_0"]
        nii_segmented = nib.Nifti1Image(array_nuevo3, img_affine)
        nib.save(nii_segmented, os.path.join(path_resultados,sample+".nii"))
    return(path)

def get_train_images(sample):
    subvolumes = next(os.walk(os.path.join(paths["SUBVOLUME_FOLDER"], sample)))[2]
    subvolumes = natsorted(subvolumes)
    masks = next(os.walk(os.path.join(paths["SUBVOLUME_MASK_FOLDER"], sample)))[2]
    masks = natsorted(masks)
    tries = 1
    for subvolume, msk in zip(natsorted(subvolumes),natsorted(masks)):
        dict_data = load(os.path.join(paths["SUBVOLUME_FOLDER"], sample, subvolume))
        volume = dict_data['arr_0']
        dict_data = load(os.path.join(paths["SUBVOLUME_MASK_FOLDER"], sample, msk))
        mask = dict_data['arr_0']
        for z in range(volume.shape[2]):
            
            Y = mask[:,:,z]
            #area = volume.shape[0]*volume.shape[1]
            #bgrd_ratio = np.sum(Y==1)/area
            #if bgrd_ratio >= back_threshold:
            X = volume[:,:,z]
            savez_compressed(os.path.join(paths["IMAGES_FOLDER"], sample, 'image'+str(tries)), X.astype(np.float32))
            savez_compressed(os.path.join(paths["IMAGES_MASK_FOLDER"], sample, 'mask'+str(tries)), Y.astype(np.float32))
            tries +=1

def get_test_images(sample):
    subvolumes = next(os.walk(os.path.join(paths["SUBVOLUME_FOLDER"], sample)))[2]
    subvolumes = natsorted(subvolumes)
    masks = next(os.walk(os.path.join(paths["SUBVOLUME_MASK_FOLDER"], sample)))[2]
    masks = natsorted(masks)
    tries = 1
    for subvolume, msk in zip(natsorted(subvolumes),natsorted(masks)):
        dict_data = load(os.path.join(paths["SUBVOLUME_FOLDER"], sample, subvolume))
        volume = dict_data['arr_0']
        dict_data = load(os.path.join(paths["SUBVOLUME_MASK_FOLDER"], sample, msk))
        mask = dict_data['arr_0']
        for z in range(volume.shape[2]):
            
            Y = mask[:,:,z]
            X = volume[:,:,z]
            savez_compressed(os.path.join(paths["IMAGES_FOLDER"], sample, 'image'+str(tries)), X.astype(np.float32))
            savez_compressed(os.path.join(paths["IMAGES_MASK_FOLDER"], sample, 'mask'+str(tries)), Y.astype(np.float32))
            tries +=1

def get_images_prueba(sample):
    """Esta funcion recibe el nombre de una muestra (carpeta) y genera los sub-volumenes.
    SOLO PARA TEST!!

    NOTA: el stride de los volumenes de test es mas grande 
    debido a que genera muchas mas imagenes y no tengo espacio.

    :param sample: nombre de la carpeta que contiene el volumen y su mascara
    :type sample: path
    """
    img = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, 1))
    img_mask = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, 2))
    img = nib.load(img)
    img_mask = nib.load(img_mask)
    image = img.get_fdata()
    image_mask = img_mask.get_fdata()

    tries = 1
    for z in range(image.shape[2]):
        Y = image_mask[:,:,z]
        X = image[:,:,z]
        savez_compressed(os.path.join(paths["IMAGES_FOLDER_PRUEBA"], sample, 'image'+str(tries)), X.astype(np.float32))
        savez_compressed(os.path.join(paths["IMAGES_MASK_FOLDER_PRUEBA"], sample, 'mask'+str(tries)), Y.astype(np.float32))
        tries +=1



def get_brain_volumes(sample):
    """Esta funcion recibe el nombre de una muestra (carpeta) y genera la extraccion del cerebro

    :param sample: nombre de la carpeta que contiene el volumen y su mascara
    :type sample: path
    """
    img = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, 1))
    img_mask = os.path.join(paths["RESULTS_3D"], sample+".nii")
    img = nib.load(img)
    img_mask = nib.load(img_mask)
    image = img.get_fdata()
    image_mask = img_mask.get_fdata()
    print(image.shape)
    for z in range (image.shape[2]):
        for y in range (image.shape[1]):
            for x in range (image.shape[0]):
                if image_mask[x,y,z] == 1:
                    image[x,y,z] = image[x,y,z]
                else:
                    image[x,y,z]=0
    
    dict_data = load(os.path.join(paths["ROTATION_MATRIX"],sample,sample+".npz"))
    img_affine = dict_data["arr_0"]
    image = image.astype(np.float32)
    img_affine = img_affine.astype(np.float32)
    nii_segmented = nib.Nifti1Image(image, img_affine)
    nib.save(nii_segmented, os.path.join(paths["RESULTS_3D"],sample+"_brain"+".nii"))
# Procesamiento de los volumenes de entrenamiento y test
# Se usa multiprocesamiento porque toma mucho tiempo
if __name__ == '__main__':
    start = timer()
    with Pool(6) as pool:
        # Comentar las lineas de abajo dependiendo de si quiere generar los volumenes o no
        #print("Sub-volumes")
        #print("Train")
        #pool.map(get_sub_volumes_train, train_files)
        #print("Test")
        #pool.map(get_sub_volumes_test, test_files)
        print("Images")
        print("Train")
        pool.map(get_train_images, train_files)
        print("Test")
        pool.map(get_test_images, test_files)

        #pool.map(get_images_prueba,train_files)
        #pool.map(get_images_prueba,test_files)
        print("finished")
    end = timer()
    
    print("Elapsed time {}".format(end-start))