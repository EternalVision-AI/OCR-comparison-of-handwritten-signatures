from signver.extractor import BaseExtractor, BaseExtractor_rev2


class MetricExtractor(BaseExtractor):
    def __init__(self, model_type="metric", batch_size=64):
        BaseExtractor.__init__(self, model_type=model_type, batch_size=64)

class MetricExtractor_rev2(BaseExtractor_rev2):
    def __init__(self, model_type="resnet18_wd4", num_features=512):
        BaseExtractor_rev2.__init__(self, model_type=model_type, num_features=num_features)
