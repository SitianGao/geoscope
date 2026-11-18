import hashlib
from PIL import Image, ImageChops


class ToolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def binary(mask):
    if mask.mode not in ('L', '1'):
        raise ToolError('invalid_arguments', 'Mask must be a single channel binary image')
    mask = mask.convert('L')
    values = {i for i, n in enumerate(mask.histogram()) if n}
    if not values <= {0, 255}:
        raise ToolError('invalid_arguments', 'Mask encoding must be 0/255')
    return mask


def scope_mask(size, scope='all', roi=None):
    w, h = size
    boxes = {'all': (0, 0, w, h), 'left_half': (0, 0, w//2, h),
             'right_half': (w//2, 0, w, h), 'top_half': (0, 0, w, h//2),
             'bottom_half': (0, h//2, w, h)}
    if scope == 'roi':
        if not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi):
            raise ToolError('invalid_arguments', 'roi must be four integer xyxy coordinates')
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ToolError('invalid_arguments', 'ROI must be nonempty and inside the original image')
        box = tuple(roi)
    elif scope in boxes and roi is None:
        box = boxes[scope]
    else:
        raise ToolError('invalid_arguments', 'Unknown scope or conflicting ROI')
    if box[0] == box[2] or box[1] == box[3]:
        raise ToolError('invalid_arguments', 'Scope is empty at this image size')
    result = Image.new('L', size, 0)
    result.paste(255, box)
    return result


def area(mask):
    return mask.histogram()[255]


def apply_scope(mask, scope='all', roi=None):
    return ImageChops.multiply(binary(mask), scope_mask(mask.size, scope, roi))


def region_statistics(mask, scope='all', roi=None):
    mask = binary(mask)
    region = scope_mask(mask.size, scope, roi)
    count = area(ImageChops.multiply(mask, region))
    total = mask.width * mask.height
    return {'target_pixels_in_scope': count, 'image_pixels': total,
            'scope_pixels': area(region), 'ratio_of_full_image': count/total,
            'ratio_within_scope': count/area(region), 'scope': scope, 'roi': roi,
            'width': mask.width, 'height': mask.height, 'unit': 'pixel',
            'mask_hash': hashlib.sha256(mask.tobytes()).hexdigest(),
            'coordinate_rule': 'original pixels, xyxy half-open; odd remainder belongs to right/bottom'}
