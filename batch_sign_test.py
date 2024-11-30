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


def batch_only_signature_matcher(reference_folder_path, match_folder_path,
                                 extractor_model_path="models/extractor/metric",
                                 cleaner_model_path="models/cleaner/small"):
    """
    Compare already extracted signatures stored in separate files
    """

    extractor = MetricExtractor()
    extractor.load(extractor_model_path)

    cleaner = Cleaner()
    cleaner.load(cleaner_model_path)

    ext = ['jpg', 'JPG', 'png', 'PNG']

    reference_files = []
    [reference_files.extend(glob.glob(f"{reference_folder_path}/*.{e}")) for e in ext]
    print(f"Total reference files: {len(reference_files)}")

    results = {}
    for reference_path in tqdm(reference_files):
        true_sign = cv2.imread(reference_path)

        files = []
        [files.extend(glob.glob(f"{match_folder_path}/*.{e}")) for e in ext]
        print(f"Total match files: {len(files)}")

        cur_result = {}
        for path in files:
            print(f"\npath = {path}")
            check_sign = cv2.imread(path)

            signatures = [true_sign, check_sign]
            # plot_np_array(signatures, fig_size=(12, 14), ncols=2, title="Extracted Signatures")

            sigs = [resnet_preprocess(x, resnet=False, invert_input=False) for x in signatures]
            # plot_np_array(sigs, "Preprocessed Signatures")

            norm_sigs = [x * (1. / 255) for x in sigs]
            # plot_np_array(norm_sigs, "Preprocessed Signatures")
            cleaned_sigs = cleaner.clean(np.array(norm_sigs))
            # plot_np_array(list(cleaned_sigs), "Cleaned Signatures")

            cleaned_feats = extractor.extract(cleaned_sigs)
            c_feat1, c_feat2 = cleaned_feats[0, :], cleaned_feats[1, :]

            matcher = Matcher()
            distance = matcher.cosine_distance(c_feat1, c_feat2)
            name = Path(path).name
            cur_result[name] = distance
            # print("Distance between Signatures = ", matcher.cosine_distance(c_feat1, c_feat2))
            # print("Is Signature 1 and 2 from same user ? -> ", matcher.verify(c_feat1, c_feat2, threshold=0.3))
        sorted_dict = dict(sorted(cur_result.items()))
        ref_name = Path(reference_path).name
        results[f'reference: {ref_name}'] = sorted_dict

    for key, value in results.items():
        print(f'{key}   ************************************** ')
        for k, v in value.items():
            print(f"{k}: {v}")
        print("*************************************\n")


def batch_signature_matcher(reference_folder_path, match_folder_path,
                            detector_model_path="models/detector/detection.onnx",
                            extractor_model_path="models/extractor/metric",
                            cleaner_model_path="models/cleaner/small"):
    """
    Compare signatures within the document contexts. Before comparison, signatures are extracted from the documents
    """

    detector = Detector()
    detector.load(detector_model_path)

    extractor = MetricExtractor()
    extractor.load(extractor_model_path)

    cleaner = Cleaner()
    cleaner.load(cleaner_model_path)

    ext = ['jpg', 'JPG', 'png', 'PNG']

    reference_files = []
    [reference_files.extend(glob.glob(f"{reference_folder_path}/*.{e}")) for e in ext]
    print(f"Total reference files: {len(reference_files)}")

    results = {}
    for reference_path in tqdm(reference_files):

        true_detections = detector.detect(reference_path)
        true_sign = true_detections[0]['image']

        files = []
        [files.extend(glob.glob(f"{match_folder_path}/*.{e}")) for e in ext]
        print(f"Total match files: {len(files)}")

        cur_result = {}
        for path in files:
            print(f"\npath = {path}")
            detections = detector.detect(path)
            check_sign = detections[0]['image']

            signatures = [true_sign, check_sign]
            # plot_np_array(signatures, fig_size=(12, 14), ncols=2, title="Extracted Signatures")

            sigs = [resnet_preprocess(x, resnet=False, invert_input=False) for x in signatures]
            # plot_np_array(sigs, "Preprocessed Signatures")

            norm_sigs = [x * (1. / 255) for x in sigs]
            # plot_np_array(norm_sigs, "Preprocessed Signatures")
            cleaned_sigs = cleaner.clean(np.array(norm_sigs))
            # plot_np_array(list(cleaned_sigs), "Cleaned Signatures")

            cleaned_feats = extractor.extract(cleaned_sigs)
            c_feat1, c_feat2 = cleaned_feats[0, :], cleaned_feats[1, :]

            matcher = Matcher()
            distance = matcher.cosine_distance(c_feat1, c_feat2)
            name = Path(path).name
            cur_result[name] = distance
            # print("Distance between Signatures = ", matcher.cosine_distance(c_feat1, c_feat2))
            # print("Is Signature 1 and 2 from same user ? -> ", matcher.verify(c_feat1, c_feat2, threshold=0.3))
        sorted_dict = dict(sorted(cur_result.items()))
        ref_name = Path(reference_path).name
        results[f'reference: {ref_name}'] = sorted_dict

    for key, value in results.items():
        print(f'{key}   ************************************** ')
        for k, v in value.items():
            print(f"{k}: {v}")
        print("*************************************\n")


def batch_only_signature_matcher_rev2(reference_folder_path, match_folder_path,
                                      extractor_model_path="models/extractor/model_epoch=115-val_loss=0.079070.pt",
                                      cleaner_model_path="models/cleaner/small"):
    """
    Rev2 of function with new trained model. Optimal threshold for the model is 10.5
    Compare already extracted signatures stored in separate files
    """

    extractor = MetricExtractor_rev2()
    extractor.load(extractor_model_path)

    cleaner = Cleaner()
    cleaner.load(cleaner_model_path)

    ext = ['jpg', 'png']

    reference_files = []
    [reference_files.extend(glob.glob(f"{reference_folder_path}/*.{e}")) for e in ext]
    print(f"Total reference files: {len(reference_files)}")

    results = {}
    for reference_path in tqdm(reference_files):
        true_sign = cv2.imread(reference_path)

        ext = ['jpg', 'JPG', 'png', 'PNG']
        files = []
        [files.extend(glob.glob(f"{match_folder_path}/*.{e}")) for e in ext]
        print(f"Total match files: {len(files)}")

        cur_result = {}
        for path in files:
            print(f"\npath = {path}")
            check_sign = cv2.imread(path)

            signatures = [true_sign, check_sign]
            # plot_np_array(signatures, fig_size=(12, 14), ncols=2, title="Extracted Signatures")

            sigs = [resnet_preprocess(x, resnet=False, invert_input=False) for x in signatures]
            # plot_np_array(sigs, "Preprocessed Signatures")

            norm_sigs = [x * (1. / 255) for x in sigs]
            # plot_np_array(norm_sigs, "Preprocessed Signatures")
            cleaned_sigs = (cleaner.clean(np.array(norm_sigs)) * 255).astype(np.uint8)
            # plot_np_array(list(cleaned_sigs), "Cleaned Signatures")

            cleaned_feats = extractor.extract(cleaned_sigs)
            c_feat1, c_feat2 = cleaned_feats[0, :], cleaned_feats[1, :]

            matcher = Matcher()
            distance = matcher.pairwise_distance(c_feat1, c_feat2)

            name = Path(path).name
            cur_result[name] = distance
            # print("Distance between Signatures = ", matcher.cosine_distance(c_feat1, c_feat2))
            # print("Is Signature 1 and 2 from same user ? -> ", matcher.verify(c_feat1, c_feat2, threshold=0.3))
        sorted_dict = dict(sorted(cur_result.items()))
        ref_name = Path(reference_path).name
        results[f'reference: {ref_name}'] = sorted_dict

    for key, value in results.items():
        print(f'{key}   ************************************** ')
        for k, v in value.items():
            print(f"{k}: {v}")
        print("*************************************\n")


def batch_signature_matcher_rev2(reference_folder_path, match_folder_path,
                                 detector_model_path="models/detector/detection.onnx",
                                 extractor_model_path="models/extractor/model_epoch=115-val_loss=0.079070.pt",
                                 cleaner_model_path="models/cleaner/small"):
    """
    Rev2 of function with new trained model. Optimal threshold for the model is 9.5
    Compare signatures within the document contexts. Before comparison, signatures are extracted from the documents
    """

    detector = Detector()
    detector.load(detector_model_path)

    extractor = MetricExtractor_rev2()
    extractor.load(extractor_model_path)

    cleaner = Cleaner()
    cleaner.load(cleaner_model_path)

    ext = ['jpg', 'JPG', 'png', 'PNG']

    reference_files = []
    [reference_files.extend(glob.glob(f"{reference_folder_path}/*.{e}")) for e in ext]
    print(f"Total reference files: {len(reference_files)}")

    results = {}
    for reference_path in tqdm(reference_files):

        true_detections = detector.detect(reference_path)
        true_sign = true_detections[0]['image']

        files = []
        [files.extend(glob.glob(f"{match_folder_path}/*.{e}")) for e in ext]
        print(f"Total match files: {len(files)}")

        cur_result = {}
        for path in files:
            print(f"\npath = {path}")
            detections = detector.detect(path)
            check_sign = detections[0]['image']

            signatures = [true_sign, check_sign]
            # plot_np_array(signatures, fig_size=(12, 14), ncols=2, title="Extracted Signatures")

            sigs = [resnet_preprocess(x, resnet=False, invert_input=False) for x in signatures]
            # plot_np_array(sigs, "Preprocessed Signatures")

            norm_sigs = [x * (1. / 255) for x in sigs]
            # plot_np_array(norm_sigs, "Preprocessed Signatures")
            cleaned_sigs = (cleaner.clean(np.array(norm_sigs)) * 255).astype(np.uint8)
            # plot_np_array(list(cleaned_sigs), "Cleaned Signatures")

            cleaned_feats = extractor.extract(cleaned_sigs)
            c_feat1, c_feat2 = cleaned_feats[0, :], cleaned_feats[1, :]

            matcher = Matcher()
            distance = matcher.pairwise_distance(c_feat1, c_feat2)
            name = Path(path).name
            cur_result[name] = distance
            # print("Distance between Signatures = ", matcher.cosine_distance(c_feat1, c_feat2))
            # print("Is Signature 1 and 2 from same user ? -> ", matcher.verify(c_feat1, c_feat2, threshold=0.3))
        sorted_dict = dict(sorted(cur_result.items()))
        ref_name = Path(reference_path).name
        results[f'reference: {ref_name}'] = sorted_dict

    for key, value in results.items():
        print(f'{key}   ************************************** ')
        for k, v in value.items():
            print(f"{k}: {v}")
        print("*************************************\n")


if __name__ == "__main__":
    # reference_folder_path = "../data/Consistent signature"
    # match_folder_path = "../data/Inconsistent signature"
    # # match_folder_path = "../data/Consistent signature"
    # # batch_signature_matcher(reference_folder_path=reference_folder_path, match_folder_path=match_folder_path)
    # batch_signature_matcher_rev2(reference_folder_path=reference_folder_path, match_folder_path=match_folder_path)

    reference_folder_path = "../datasets/signatures/original"
    # match_folder_path = "../datasets/signatures/original"
    match_folder_path = "../datasets/signatures/forgery"
    # batch_only_signature_matcher(reference_folder_path=reference_folder_path, match_folder_path=match_folder_path)
    batch_only_signature_matcher_rev2(reference_folder_path=reference_folder_path, match_folder_path=match_folder_path)
