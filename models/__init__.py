from .lowresresnet9 import lowres_resnet9
from .openlth_resnet import ResNet

from functools import partial

all_classifiers = {
    "lowres_resnet9": lowres_resnet9,
}


def get_model(name):
    if name.startswith("openlth_"):
        return partial(ResNet.get_model_from_name, name=name.replace("openlth_", ""))
    else:
        return all_classifiers.get(name)