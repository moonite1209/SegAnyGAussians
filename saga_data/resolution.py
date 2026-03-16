_WARNED = False


def compute_target_size(orig_w: int, orig_h: int, resolution: int, resolution_scale: float):
    global _WARNED
    if resolution in [1, 2, 4, 8]:
        scale = resolution * resolution_scale
    else:
        if resolution == -1:
            if orig_w > 1600:
                if not _WARNED:
                    print(
                        "[ INFO ] Encountered large images (>1.6K width), rescaling to 1.6K.\n"
                        "If this is not desired, set --resolution/-r to 1"
                    )
                    _WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution
        scale = float(global_down) * float(resolution_scale)

    target_w = int(orig_w / scale)
    target_h = int(orig_h / scale)
    return target_w, target_h
