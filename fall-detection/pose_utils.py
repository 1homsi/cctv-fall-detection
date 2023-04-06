import numpy as np


'''
    The normalize points function is used to normalize the points to (0-1) scale.
    it takes the points and the width and height of the image and returns the normalized points.
    xy is the points to be normalized
    width is the width of the image and height is the height of the image
    flip is used to flip the points horizontally if the image is flipped but the points are not. in this case it is 
    set to false by default unless it is set to true.
''' 

def normalize_points_with_size(xy, width, height, flip=False):
    """
    Normalize scale points in image with size of image to (0-1).
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:    # (parts, xy)
        xy = np.expand_dims(xy, 0)  # (1, parts, xy)
    xy[:, :, 0] /= width    # (frames, parts, xy)
    xy[:, :, 1] /= height   # (frames, parts, xy)
    if flip:
        xy[:, :, 0] = 1 - xy[:, :, 0]   # (frames, parts, xy)
    return xy   # (frames, parts, xy)


'''
    the scale pose function is used to scale the points to (-1,1) scale.
    it takes the points and returns the scaled points.  
'''
def scale_pose(xy):
    """Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()
