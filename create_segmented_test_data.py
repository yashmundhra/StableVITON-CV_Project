import os
import shutil


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


def get_relevant_test_ids():
    # reads a txt file
    with open(os.path.join(os.getcwd(), 'DATA', 'test_pairs.txt'), 'r') as f:
        image_ids = []
        for line in f.readlines():
            person = line.split(' ')[0]
            garment = line.split(' ')[1]
            # remove newline from garment
            garment = garment[:-1]
            image_ids.extend([person, garment])
    image_ids = [image_id[:-4] for image_id in image_ids]
    return image_ids


def copy_relevant_test_files(src_folder, dest_folder, image_ids):
    path = os.path.join(os.getcwd(), "DATA", "zalando-hd-resized")
    src_path = os.path.join(path, src_folder)
    dest_path = os.path.join(path, dest_folder)
    os.makedirs(dest_path, exist_ok=True)

    subfolders = [f.path for f in os.scandir(src_path) if f.is_dir()]

    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)

        new_subfolder_path = os.path.join(dest_path, folder_name)
        os.makedirs(new_subfolder_path, exist_ok=True)

        all_items = os.listdir(subfolder)
        items = []
        for image_id in image_ids:
            items.extend([item for item in all_items if item.startswith(image_id)])

        items = [os.path.join(subfolder, item) for item in items]

        for item in items:
            if os.path.isfile(item):
                shutil.copy(item, new_subfolder_path)


images_ids = get_relevant_test_ids()
copy_relevant_test_files("test", "test_segment", images_ids)  # from each subfolder in test

# Copy newly segmented files to the corresponding folders
src_folder = "masks"
dest_folder_mask = "test_segment/agnostic-mask"
dest_folder_non_mask = "test_segment/agnostic-v3.2"
copy_files_based_on_name(src_folder, dest_folder_mask, dest_folder_non_mask)
