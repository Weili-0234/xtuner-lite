# Copyright (c) OpenMMLab. All rights reserved.
from .layer_decay_optim_wrapper_constructor import LearningRateDecayOptimWrapperConstructor
from .utils import get_layer_depth_for_CLIPVisionModel
__all__ = [
    'LearningRateDecayOptimWrapperConstructor', 'get_layer_depth_for_CLIPVisionModel'
]