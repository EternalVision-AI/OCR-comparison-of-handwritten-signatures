from signver.detector import Detector
from signver.cleaner import Cleaner
from signver.extractor import MetricExtractor
from signver.matcher import Matcher
from signver.utils import data_utils
from signver.utils.data_utils import resnet_preprocess
from signver.utils.visualization_utils import plot_np_array, visualize_boxes, get_image_crops, make_square

import numpy as np


detector_model_path = "models/detector/detection.onnx"
detector = Detector()
detector.load(detector_model_path)

extractor_model_path = "models/extractor/metric"
extractor = MetricExtractor()
extractor.load(extractor_model_path)

cleaner_model_path = "models/cleaner/small"
cleaner = Cleaner()
cleaner.load(cleaner_model_path)


# img1_path = "/home/iv/PycharmProjects/Puzik/Signature/signver-main/data/test_2/Consistent signature/3_page-0001_cleaned.jpg"
# img2_path = "/home/iv/PycharmProjects/Puzik/Signature/signver-main/data/test_2/Inconsistent signature/4_page-0001_cleaned.jpg"
# img3_path = "/home/iv/PycharmProjects/Puzik/Signature/signver-main/data/test_2/Inconsistent signature/5_page-0001_cleaned.jpg"

true_sign_path = "/home/iv/PycharmProjects/Puzik/Signature/signver-main/data/test_2/Consistent signature/1_page-0001.jpg"
# check_doc_path = "/home/iv/PycharmProjects/Puzik/Signature/data/Consistent signature/1_page-0001.jpg"
check_doc_path = "/home/iv/PycharmProjects/Puzik/Signature/data/Inconsistent signature/3_page-0001.jpg"

true_sign_np = data_utils.img_to_np_array(true_sign_path)
# inverted_image1_np = data_utils.img_to_np_array(true_sign_path, invert_image=True)

detections = detector.detect(check_doc_path)
check_sign = detections[0]['image']

signatures = [true_sign_np, check_sign]
# plot_np_array(signatures, fig_size=(12, 14), ncols=2, title="Extracted Signatures")

# image1_np = data_utils.img_to_np_array(img1_path)
# inverted_image1_np = data_utils.img_to_np_array(img1_path, invert_image=True)
# img1_tensor = tf.convert_to_tensor(inverted_image1_np)
# img1_tensor = img1_tensor[tf.newaxis, ...]

# image2_np = data_utils.img_to_np_array(img2_path)
# inverted_image2_np = data_utils.img_to_np_array(img2_path, invert_image=True)
# img2_tensor = tf.convert_to_tensor(inverted_image2_np)
# img2_tensor = img2_tensor[tf.newaxis, ...]

# image3_np = data_utils.img_to_np_array(img3_path)
# inverted_image3_np = data_utils.img_to_np_array(img3_path, invert_image=True)
# img3_tensor = tf.convert_to_tensor(inverted_image3_np)
# img3_tensor = img3_tensor[tf.newaxis, ...]

# signatures = [inverted_image1_np, inverted_image2_np, inverted_image3_np]
# plot_np_array(signatures, fig_size=(12, 14), ncols=3, title="Extracted Signatures")

sigs = [resnet_preprocess(x, resnet=False, invert_input=False) for x in signatures]
plot_np_array(sigs, "Preprocessed Signatures")
#
# test_line = data_utils.img_to_np_array("data/test/extractor/test_lines.png")
# test_line = resnet_preprocess(test_line, resnet=False, invert_input=False)
#
# cn = cleaner.clean(np.expand_dims(test_line, axis=0))
#
norm_sigs = [x * (1. / 255) for x in sigs]
# plot_np_array(norm_sigs, "Preprocessed Signatures")
cleaned_sigs = cleaner.clean(np.array(norm_sigs))
plot_np_array(list(cleaned_sigs), "Cleaned Signatures")

feats = extractor.extract(np.array(sigs) / 255)
# feat1, feat2, feat3 = feats[0, :], feats[1, :], feats[2, :]
feat1, feat2 = feats[0, :], feats[1, :]

cleaned_feats = extractor.extract(cleaned_sigs)
# c_feat1, c_feat2, c_feat3 = cleaned_feats[0, :], cleaned_feats[1, :], cleaned_feats[2, :]
c_feat1, c_feat2 = cleaned_feats[0, :], cleaned_feats[1, :]

matcher = Matcher()
# print("Distance between Signature 1 and 1 -> ", matcher.cosine_distance(feat1, feat1))
print("Distance between Signature 1 and 2 -> ", matcher.cosine_distance(feat1, feat2))
# print("Distance between Signature 1 and 3 -> ", matcher.cosine_distance(feat1, feat3))
# print("Distance between Signature 2 and 3 -> ", matcher.cosine_distance(feat2, feat3))

# print("Is Signature 1 and 1 from same user ? -> ", matcher.verify(feat1, feat1, threshold=0.3))
# print("Is Signature 1 and 2 from same user ? -> ", matcher.verify(feat1, feat2, threshold=0.18))
# print("Is Signature 1 and 3 from same user ? -> ", matcher.verify(feat1, feat3, threshold=0.18))
# print("Is Signature 2 and 3 from same user ? -> ", matcher.verify(feat2, feat3, threshold=0.18))

# matcher = Matcher()
# print("c_feat Distance between Signature 1 and 1 -> ", matcher.cosine_distance(c_feat1, c_feat1))
print("c_feat Distance between Signature 1 and 2 -> ", matcher.cosine_distance(c_feat1, c_feat2))
# print("Distance between Signature 1 and 3 -> ", matcher.cosine_distance(c_feat1, c_feat3))
# print("Distance between Signature 2 and 3 -> ", matcher.cosine_distance(c_feat2, c_feat3))
#
# print("Is Signature 1 and 1 from same user ? -> ", matcher.verify(c_feat1, c_feat1, threshold=0.3))
# print("Is Signature 1 and 2 from same user ? -> ", matcher.verify(c_feat1, c_feat2, threshold=0.3))
# print("Is Signature 1 and 3 from same user ? -> ", matcher.verify(c_feat1, c_feat3, threshold=0.3))
# print("Is Signature 2 and 3 from same user ? -> ", matcher.verify(c_feat2, c_feat3, threshold=0.3))
