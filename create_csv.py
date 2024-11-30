import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def create_csv(csv_path, dataset_folder):
    """
    Create CSV file from a signature dataset folder
    dataset_folder - path to a dataset folder with structure, like: "name", "name_forg",
        where "name"       - is the name of the folder with original signatures
              "name_forg"  - is the name of the folder with forgery signatures

    csv_path - path to a target CSV file
    """

    ext = ['png', 'PNG', 'jpg']
    dirs = glob.glob(f"{dataset_folder}/*", recursive=False)
    count = 0
    csv_file = open(csv_path, "a")
    for dir in dirs:
        dir_name = Path(dir).name
        if '_forg' not in dir_name:
            print("*****************************************************************************************")
            print(f'dir = {dir}')
            true_path = f'{dataset_folder}/{dir_name}'
            false_path = f'{dataset_folder}/{dir_name}_forg'
            true_files = []
            [true_files.extend(glob.glob(f"{true_path}/*.{e}")) for e in ext]
            false_files = []
            [false_files.extend(glob.glob(f"{false_path}/*.{e}")) for e in ext]

            for i in range(len(true_files)):
                anchor_path = true_files[i]
                for j in range(len(true_files)):
                    if i == j:
                        continue
                    true_path = true_files[j]
                    csv_file.write(f'{anchor_path},{true_path},1\n')
                    count += 1
                for j in range(len(false_files)):
                    false_path = false_files[j]
                    csv_file.write(f'{anchor_path},{false_path},0\n')
                    count += 1

    print('count = ', count)
    csv_file.close()


if __name__ == "__main__":
    csv_path = '.../datasets/Signature_Verification_Dataset/test_customer_label.csv'
    dataset_folder = '../datasets/Signature_Verification_Dataset/test_customer'

    create_csv(csv_path, dataset_folder)