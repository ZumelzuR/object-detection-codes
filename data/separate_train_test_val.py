import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from random import sample
from shutil import copy2


dest_train_path = "images/train"
dest_test_path = "images/test"
dest_val_path = "images/val"

# total imagenes 7947
#train 5960 (75%)
# entre jtl y positivos
len_train = 5960
#val 1192 (15%)
len_val = 1192
#test 795 (10%)
len_positivos_test=490
len_negativos_test=305


#utils
#listar archivos de un directorio con una cierta extension
def list_files1(directory, extension):
    return list(f for f in os.listdir(directory) if f.endswith('.' + extension))

def list_files_contain(directory, value):
    return list(f for f in os.listdir(directory) if f.find(value)!=-1)

#contar elementos de un directorio
def filecount(dir_name):
    return len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])

#TRANSFORMAR path de imagenes ya que positivos empiezan con jpg mientras que el resto con JPEG
def transform_image_path(xml_path):
    image_id,extension=xml_path.split(".");
    if image_id.find("positivos") == 0:
        return image_id+".jpg"
    else:
        return image_id+".JPEG"

#crea una lista con el nombre de cada elemento por cada conjunto train test val
# esto sirve para crear los dataset de yolo
def create_list_train_test(extension):
    test_files=list_files1(os.path.join(os.getcwd(), dest_test_path),"xml")
    val_files=list_files1(os.path.join(os.getcwd(), dest_val_path),"xml")
    train_files=list_files1(os.path.join(os.getcwd(),dest_train_path),"xml")

    file_test = open(os.path.join(os.getcwd(), 'test'+extension+'.txt'), 'w')
    file_val = open(os.path.join(os.getcwd(), 'val'+extension+'.txt'), 'w')
    file_train = open(os.path.join(os.getcwd(), 'train'+extension+'.txt'), 'w')

    for train in train_files:
        file_train.write("%s\n" % train.split(".")[0])
    for val in val_files:
        file_val.write("%s\n" % val.split(".")[0])
    for test in test_files:
        file_test.write("%s\n" % test.split(".")[0])

    print("Successfully created lists of test, val and train objects")

#separa el conjunto de annotations e imagenes en train test y val
#(copia las imagenes y xml en carpetas train test val
def generate_dataset_adding_negatives():
    xml_path = os.path.join(os.getcwd(), 'Annotations')
    image_path = os.path.join(os.getcwd(), 'JPEGImages')
    neg_path = os.path.join(os.getcwd(), 'conjSLRatCrop/negativos1')

    total_paths = os.listdir(xml_path)
    total_indx= range(0, len(total_paths)-1)

    positives_paths = list_files_contain(xml_path,"positivos")
    positives_indx = range(0, len(positives_paths) - 1)

    #creo el conjunto de test (10%)
    # saco solo len_negativos_test ejemplos de la carpeta negativos1
    negatives_paths = os.listdir(neg_path)
    neg_paths_test = [negatives_paths[i] for i in range(0,len_negativos_test)];

    #saco al azar len_positivos_test ejemplos de los positivos que tengo
    pos_indx_test = sample(positives_indx, len_positivos_test)
    pos_paths_test = [positives_paths[i] for i in pos_indx_test];

    for i_test_n in neg_paths_test:
        #copio solo imagenes ya que no necesito anotaciones para los negativos
        copy2(os.path.join(neg_path, i_test_n), os.path.join(os.getcwd(), dest_test_path))
    for i_test_p in pos_paths_test:
        #copio xml e imagenes positivas
        copy2(os.path.join(xml_path, i_test_p), os.path.join(os.getcwd(), dest_test_path))
        copy2(os.path.join(image_path, transform_image_path(i_test_p)), os.path.join(os.getcwd(), dest_test_path))

    #Creo el conjunto de training y de validacion
    # eliminamos de la lista los positivos que seleccionamos para test
    temp_paths = np.delete(total_paths, pos_indx_test)
    temp_indx = range(0, len(temp_paths) - 1)

    #validación
    val_indx = sample(temp_indx, len_val)
    val_paths = [temp_paths[i] for i in val_indx]

    #training
    # eliminamos los seleccionados para val
    rest_paths = np.delete(temp_paths, val_indx)
    rest_indx = range(0, len(rest_paths) - 1)
    # de los que queden obtengo len_train objetos al azar
    train_indx = sample(rest_indx, len_train)
    train_paths = [rest_paths[i] for i in train_indx]


    for i_train in train_paths:
        #por cada objeto de train, copio el xml y su imagen en la carpeta train
        copy2(os.path.join(xml_path, i_train), os.path.join(os.getcwd(), dest_train_path))
        copy2(os.path.join(image_path, transform_image_path(i_train)), os.path.join(os.getcwd(), dest_train_path))
    for i_val in val_paths:
        # por cada objeto de val, copio el xml y su imagen en la carpeta val
         copy2(os.path.join(xml_path, i_val), os.path.join(os.getcwd(), dest_val_path))
         copy2(os.path.join(image_path, transform_image_path(i_val)), os.path.join(os.getcwd(), dest_val_path))

    print('Successfully separated images in train,test and val')

def generate_dataset():
    xml_path = os.path.join(os.getcwd(), 'Annotations')
    image_path = os.path.join(os.getcwd(), 'JPEGImages')

    jtl_paths = list_files_contain(xml_path,"jtl")
    jtl_indx= range(0, len(jtl_paths)-1)

    positives_paths = list_files_contain(xml_path,"positivos")
    positives_indx = range(0, len(positives_paths) - 1)

    #creo el conjunto de test (10%)

    len_test=len_positivos_test+len_negativos_test

    #saco al azar len_positivos_test ejemplos de los positivos que tengo
    indx_test = sample(positives_indx, len_test)
    paths_test = [positives_paths[i] for i in indx_test];

    for i_test_p in paths_test:
        #copio xml e imagenes positivas
        copy2(os.path.join(xml_path, i_test_p), os.path.join(os.getcwd(), dest_test_path))
        copy2(os.path.join(image_path, transform_image_path(i_test_p)), os.path.join(os.getcwd(), dest_test_path))

    #Creo el conjunto de training y de validacion
    # eliminamos de la lista los positivos que seleccionamos para test
    rest_positive_paths = np.delete(positives_paths, indx_test)
   # rest_positive_indx = range(0, len(rest_positive) - 1)

    #juntamos esta sublista de positivos con los jtl
    total_path=list(rest_positive_paths)+jtl_paths
    total_indx = range(0, len(total_path) - 1)

    #validación
    val_indx = sample(total_indx, len_val)
    val_paths = [total_path[i] for i in val_indx]

    #training
    # eliminamos los seleccionados para val
    train_paths = np.delete(total_path, val_indx)

    for i_train in train_paths:
        #por cada objeto de train, copio el xml y su imagen en la carpeta train
        copy2(os.path.join(xml_path, i_train), os.path.join(os.getcwd(), dest_train_path))
        copy2(os.path.join(image_path, transform_image_path(i_train)), os.path.join(os.getcwd(), dest_train_path))
    for i_val in val_paths:
        # por cada objeto de val, copio el xml y su imagen en la carpeta val
         copy2(os.path.join(xml_path, i_val), os.path.join(os.getcwd(), dest_val_path))
         copy2(os.path.join(image_path, transform_image_path(i_val)), os.path.join(os.getcwd(), dest_val_path))

    print('Successfully separated images in train,test and val')

#generate_dataset()
create_list_train_test("_list_full")

#luego generar los csv con xml_to_csv.
#una vez esten en csv, utilizar generate_tf_record.py para pasarlos a trrecord



