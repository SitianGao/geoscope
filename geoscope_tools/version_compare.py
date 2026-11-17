from PIL import Image, ImageChops
from .statistics import ToolError, binary, area


def mask_compare(base, target):
    base, target = binary(base), binary(target)
    if base.size != target.size:
        raise ToolError('invalid_arguments', 'Mask sizes differ; implicit resampling is prohibited')
    intersection = area(ImageChops.multiply(base, target))
    union = area(ImageChops.lighter(base, target))
    added = ImageChops.subtract(target, base)
    removed = ImageChops.subtract(base, target)
    diff = Image.merge('RGB', (removed, added, Image.new('L', base.size)))
    return {'width': base.width, 'height': base.height, 'base_pixels': area(base),
            'target_pixels': area(target), 'intersection_pixels': intersection, 'union_pixels': union,
            'added_pixels': area(added), 'removed_pixels': area(removed),
            'prediction_agreement_iou': intersection/union if union else 1.0,
            'empty_set_rule': 'both_empty_agreement_is_one',
            'notice': 'Prediction agreement is not ground truth accuracy; diff red=removed, green=added'}, diff
