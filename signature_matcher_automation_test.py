from signver.detector import Detector
from signver.cleaner import Cleaner
from signver.matcher import Matcher
from signver.extractor import MetricExtractor, MetricExtractor_rev2

from signver.utils.data_utils import invert_img, resnet_preprocess
from signver.utils.visualization_utils import plot_np_array

from tqdm import tqdm
from pathlib import Path

import numpy as np
import glob
import cv2

import os

def list_files(path):
	onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.png') or f.endswith('.PNG')]
	onlyfiles.sort()
	return onlyfiles

class SignaturesMatcher:
    """
    Compare already extracted signatures stored in separate files
    """

    def __init__(self, extractor_model_path: str = "models/extractor/model_epoch=115-val_loss=0.079070.pt",
                 cleaner_model_path: str = "models/cleaner/small"):
        """
        :param extractor_model_path: Path to feature extractor model (Rev2)
        :param cleaner_model_path: Path to cleaner model
        """
        self.extractor = MetricExtractor_rev2()
        self.extractor.load(extractor_model_path)

        self.cleaner = Cleaner()
        self.cleaner.load(cleaner_model_path)

        self.matcher = Matcher()

    def compare_signatures(self, path_signature_1: str, path_signature_2: str,
                           up_distance_limit: float = 10.0, low_distance_limit: float = 6.0):
        """
        Compares 2 signatures (already extracted from a document)
        :param path_signature_1: Path to the first signature file
        :param path_signature_2: Path to the second signature file
        :param low_distance_limit: Low distance limit for a confidence evaluation. Default = 6.0. Signature pair with
                                   a distance lower than this limit will have a confidence equal to 100%.
        :param up_distance_limit: Up distance limit for a confidence evaluation. Default = 10.0. Signature pair with
                                  a distance higher than this limit will have a confidence equal to 0%.
        :return:
            similarity: Confidence score (similarity) between 2 signatures in percents
                        (0% - signatures are completely different, 100% - signatures are equal to each other)
        """
        assert up_distance_limit > low_distance_limit, "up_distance_limit should be higher then low_distance_limit"
        signature_1 = cv2.imread(path_signature_1)
        signature_2 = cv2.imread(path_signature_2)
        signatures = [signature_1, signature_2]

        sigs = [resnet_preprocess(x, resnet=False, invert_input=False) for x in signatures]

        norm_sigs = [x * (1. / 255) for x in sigs]
        cleaned_sigs = (self.cleaner.clean(np.array(norm_sigs)) * 255).astype(np.uint8)

        cleaned_feats = self.extractor.extract(cleaned_sigs)
        c_feat1, c_feat2 = cleaned_feats[0, :], cleaned_feats[1, :]

        distance = self.matcher.pairwise_distance(c_feat1, c_feat2)

        # print('distance = ', distance)

        if distance < low_distance_limit:
            similarity = 100.0
        elif distance > up_distance_limit:
            similarity = 0.0
        else:
            similarity = 100 * (up_distance_limit - distance) / (up_distance_limit - low_distance_limit)

        with open('log_forgery_file.txt', 'a') as f:
            f.write("-------------------------------------------------\n")
            f.write(f"{path_signature_1} "+" Vs "+f" {path_signature_2}\n")
            f.write(f"Confidence score (similarity) = {similarity:.2f}%\n")

        return similarity

    def compare_signature_with_folder(self, path_signature: str, path_folder: str,
                                      up_distance_limit: float = 10.0, low_distance_limit: float = 6.0):
        """
        Compares 2 signatures (already extracted from a document)
        :param path_signature: Path to a signature file
        :param path_folder: Path to a folder with signature files
        :param low_distance_limit: Low distance limit for a confidence evaluation. Default = 6.0. Signature pair with
                                   a distance lower than this limit will have a confidence equal to 100%.
        :param up_distance_limit: Up distance limit for a confidence evaluation. Default = 10.0. Signature pair with
                                  a distance higher than this limit will have a confidence equal to 0%.
        :return:
            average_similarity: Average confidence score (similarity) between a separate signature and all signatures
                                in a target folder in percents.
                                (0% - signatures are completely different, 100% - signatures are equal to each other)
        """
        assert up_distance_limit > low_distance_limit, "up_distance_limit should be higher then low_distance_limit"
        signature_1 = cv2.imread(path_signature)
        signature_1_file_name = Path(path_signature).name

        ext = ['png', 'PNG', 'jpg', 'jpeg']
        folder_signatures = []
        [folder_signatures.extend(glob.glob(f"{path_folder}/*.{e}")) for e in ext]
        similarity_results = {}
        average_similarity = 0
        max_file_len = 0

        # print(folder_signatures)

        for signature_2 in folder_signatures:
            signature_2_file_name = Path(signature_2).name
            if len(signature_2_file_name) > max_file_len:
                max_file_len = len(signature_2_file_name)
            signature_2 = cv2.imread(signature_2)
            signatures = [signature_1, signature_2]
            sigs = [resnet_preprocess(x, resnet=False, invert_input=False) for x in signatures]

            norm_sigs = [x * (1. / 255) for x in sigs]
            cleaned_sigs = (self.cleaner.clean(np.array(norm_sigs)) * 255).astype(np.uint8)

            cleaned_feats = self.extractor.extract(cleaned_sigs)
            c_feat1, c_feat2 = cleaned_feats[0, :], cleaned_feats[1, :]

            distance = self.matcher.pairwise_distance(c_feat1, c_feat2)

            if distance < low_distance_limit:
                similarity = 100.0
            elif distance > up_distance_limit:
                similarity = 0.0
            else:
                similarity = 100 * (up_distance_limit - distance) / (up_distance_limit - low_distance_limit)
            average_similarity += similarity
            similarity_results[signature_2_file_name] = similarity

        average_similarity = average_similarity / len(folder_signatures)

        with open('log_forgery_folder.txt', 'a') as f:
            f.write("-----------------------------------------------------------------------\n")
            f.write(f'Folder name: {path_folder}\n')
            f.write(f'Signature name: {path_signature}\n')
            f.write(f'Average Confidence Score (Similarity): {average_similarity:.2f} %\n')
            # f.write(f'Signature: {signature_1_file_name}\n')
            for key, value in similarity_results.items():
                f.write(f'{key:{max_file_len}s} ----- {value:.2f}%\n')


        return average_similarity


if __name__ == "__main__":
    log_original_file = 'log_original_file.txt'
    log_original_folder = 'log_original_folder.txt'
    log_forgery_file = 'log_forgery_file.txt'
    log_forgery_folder = 'log_forgery_folder.txt'

    extractor_model_path = "models/extractor/model_epoch=115-val_loss=0.079070.pt"
    cleaner_model_path = "models/cleaner/small"

    path_signature_1 = '../test/001/01_049.png'
    path_signature_2 = '../test/001/02_049.png'

    matcher = SignaturesMatcher(extractor_model_path=extractor_model_path, cleaner_model_path=cleaner_model_path)


    test_dir_path = '../test'
    
    for index in range(1,22):
        dirNum = ""
        if len(str(index)) == 1:
            dirNum = "0" + str(index)
        elif len(str(index)) == 2:
            dirNum = str(index)
        dir_1 = test_dir_path + "/0"+dirNum
        ls_files_1 = list_files(dir_1)
        dir_2 = test_dir_path + "/0"+dirNum+"_forg"
        ls_files_2 = list_files(dir_2)

        for input_file_1 in ls_files_1:
            inputFilename_1 = os.path.join(dir_1, input_file_1)

            for input_file_2 in ls_files_2:
                inputFilename_2 = os.path.join(dir_2, input_file_2)

                result = matcher.compare_signatures(path_signature_1=inputFilename_1, path_signature_2=inputFilename_2,
                                        up_distance_limit=10.0, low_distance_limit=6.0)
                print(f'Confidence score (similarity) = {result:.2f}%')

            # print("dir_2")
            # print(dir_2)
            average_result = matcher.compare_signature_with_folder(path_signature=inputFilename_1, path_folder=dir_2,
                                                           up_distance_limit=10.0, low_distance_limit=6.0)









    

    # path_folder = '../test/001_forg'
    # average_result = matcher.compare_signature_with_folder(path_signature=path_signature_1, path_folder=path_folder,
    #                                                        up_distance_limit=10.0, low_distance_limit=6.0)
    # print(f'Average confidence score (similarity) = {average_result:.2f}%')
