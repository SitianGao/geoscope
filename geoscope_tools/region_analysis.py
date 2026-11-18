from collections import deque
from .statistics import ToolError, apply_scope


def connected_region_analysis(mask, scope='all', roi=None, min_area_pixels=1):
    if type(min_area_pixels) is not int or min_area_pixels < 1:
        raise ToolError('invalid_arguments', 'min_area_pixels must be a positive integer')
    mask = apply_scope(mask, scope, roi)
    w, h = mask.size
    pixels = mask.tobytes()
    seen = bytearray(len(pixels))
    kept, filtered = [], []
    for seed, value in enumerate(pixels):
        if not value or seen[seed]:
            continue
        seen[seed] = 1
        queue = deque([seed])
        n = sx = sy = 0
        x1, y1, x2, y2 = w, h, 0, 0
        while queue:
            i = queue.popleft()
            y, x = divmod(i, w)
            n += 1; sx += x; sy += y
            x1, y1, x2, y2 = min(x1, x), min(y1, y), max(x2, x+1), max(y2, y+1)
            for yy in range(max(0, y-1), min(h, y+2)):
                for xx in range(max(0, x-1), min(w, x+2)):
                    j = yy*w+xx
                    if pixels[j] and not seen[j]:
                        seen[j] = 1; queue.append(j)
        item = {'component_id': len(kept)+len(filtered)+1, 'area_pixels': n,
                'bbox_xyxy': [x1, y1, x2, y2], 'center': [sx/n, sy/n]}
        if n < min_area_pixels:
            filtered.append({**item, 'reason': 'below_min_area_pixels'})
        else:
            kept.append(item)
    areas = [c['area_pixels'] for c in kept]
    bins = [(1, 9), (10, 99), (100, 999), (1000, None)]
    return {'connected_components_count': len(kept), 'count_before_filter': len(kept)+len(filtered),
            'count_after_filter': len(kept), 'min_area_pixels': min_area_pixels,
            'components': kept, 'filtered_components': filtered, 'connectivity': 8,
            'area_mean': sum(areas)/len(areas) if areas else 0,
            'area_min': min(areas, default=0), 'area_max': max(areas, default=0),
            'area_histogram': [{'min': lo, 'max': hi, 'count': sum(a >= lo and (hi is None or a <= hi) for a in areas)} for lo, hi in bins],
            'scope': scope, 'roi': roi, 'processing_order': 'scope_then_components',
            'notice': '8-connected candidates are not object instances; scope boundaries may truncate or split regions.'}


def spatial_relation_analysis(mask, other_mask, near_distance_pixels=10):
    if type(near_distance_pixels) not in (int, float) or not 0 <= near_distance_pixels <= 100000:
        raise ToolError('invalid_arguments', 'Invalid pixel distance')
    from .version_compare import mask_compare
    comparison, _ = mask_compare(mask, other_mask)
    a, b = mask.getbbox(), other_mask.getbbox()
    result = {'intersects': comparison['intersection_pixels'] > 0, 'unit': 'pixel',
              'direction_basis': 'image coordinates; not geographic directions'}
    if a is None or b is None:
        return {**result, 'bbox_distance_pixels': None, 'near': None, 'relative_center': None}
    import math
    distance = math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))
    return {**result, 'bbox_distance_pixels': distance, 'near': distance <= near_distance_pixels,
            'relative_center': {'horizontal': 'left' if a[0]+a[2] < b[0]+b[2] else 'right' if a[0]+a[2] > b[0]+b[2] else 'aligned',
                                'vertical': 'above' if a[1]+a[3] < b[1]+b[3] else 'below' if a[1]+a[3] > b[1]+b[3] else 'aligned'}}
