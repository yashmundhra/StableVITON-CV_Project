import os
import shutil


def copy_first_8_items(src_folder, dest_folder):
    path = os.path.join(os.getcwd(), "DATA/zalando-hd-resized/")
    src_path = os.path.join(path, src_folder)
    dest_path = os.path.join(path, dest_folder)
    os.makedirs(dest_path, exist_ok=True)

    subfolders = [f.path for f in os.scandir(src_path) if f.is_dir()]

    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)

        new_subfolder_path = os.path.join(dest_path, folder_name)
        os.makedirs(new_subfolder_path, exist_ok=True)

        items = [os.path.join(subfolder, item) for item in os.listdir(subfolder)]

        for item in items[:8]:
            if os.path.isfile(item):
                shutil.copy(item, new_subfolder_path)
            elif os.path.isdir(item):
                shutil.copytree(item, os.path.join(new_subfolder_path, os.path.basename(item)))


def copy_files_based_on_name(src_folder, dest_folder_mask, dest_folder_non_mask):
    path = os.path.join(os.getcwd(), "DATA", "zalando-hd-resized")
    src_path = os.path.join(path, src_folder)
    dest_path_agnostic = os.path.join(path, dest_folder_non_mask)
    dest_path_agnostic_mask = os.path.join(path, dest_folder_mask)

    os.makedirs(dest_path_agnostic_mask, exist_ok=True)
    os.makedirs(dest_path_agnostic, exist_ok=True)

    files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]

    for file in files:
        src_file_path = os.path.join(src_path, file)
        if 'mask' in file:
            shutil.copy(src_file_path, dest_path_agnostic_mask)
        else:
            shutil.copy(src_file_path, dest_path_agnostic)


# Copy first 8 items from the test folder to the test_segment folder
# src_folder = "test"
# dest_folder = "test_segment"
# copy_first_8_items(src_folder, dest_folder)

# Move newly segmented files to the corresponding folders
src_folder = "masks"
dest_folder_mask = "test_segment/agnostic-mask"
dest_folder_non_mask = "test_segment/agnostic-v3.2"
copy_files_based_on_name(src_folder, dest_folder_mask, dest_folder_non_mask)
