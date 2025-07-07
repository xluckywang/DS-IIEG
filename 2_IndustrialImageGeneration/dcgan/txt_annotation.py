import os

if __name__ == "__main__":
    #---------------------------------#
    # Point to the folder where the dataset is located
    # Default pointing to datasets in the root directory
    #---------------------------------#
    # datasets_path   = "../data/datasetA/lhj_train/train"
    datasets_path = "../../0_data/datasetOri/MTD/train/MT_Blowhole"
    # datasets_path = "../data/datasets_original/X-SDD/oxide_scale_of_plate_system"
    photos_names    = os.listdir(datasets_path)
    photos_names    = sorted(photos_names)

    list_file       = open('train_lines.txt', 'w')
    for photo_name in photos_names:
        if(photo_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
            list_file.write(os.path.join(os.path.abspath(datasets_path), photo_name))
            list_file.write('\n')
    list_file.close()

