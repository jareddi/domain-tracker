from tracker_utils import *

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
 Properties Functions
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""
def unwrap_1D(values, L):
    """
    Shift 1D values under periodic boundary conditions to be near each other.
    """
    values = np.array(values)
    ref = values[0]
    unwrapped = [ref]

    for val in values[1:]:
        candidates = [val + shift * L for shift in [-1, 0, 1]]
        best = min(candidates, key=lambda x: abs(x - ref))
        unwrapped.append(best)

    return np.array(unwrapped)

def compute_orientation(xs, ys, masses=None):
    """
    Compute the orientation of a 2D point cloud using PCA.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if masses is None:
        masses = np.ones_like(xs)
    else:
        masses = np.asarray(masses)

    assert xs.shape == ys.shape == masses.shape, "Input arrays must have the same shape"

    points = np.stack((xs, ys), axis=1)
    masses = masses / np.sum(masses)
    # Compute weighted centroid
    centroid = np.average(points, axis=0, weights=masses)
    
    # Centered coordinates
    centered = points - centroid
    # Weighted covariance matrix
    cov = np.einsum('i,ij,ik->jk', masses, centered, centered)
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Angle of major axis in radians
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    if angle > np.pi/2:
        angle -= np.pi
    elif angle < -np.pi/2:
        angle += np.pi

    return centroid, angle

def get_contour_properties(contour):
    """
    Takes a stitched domain and returns contour properties 
    """
    properties = {}
    M = cv.moments(contour)
    if M['m00'] == 0:
        properties['center']= [np.mean(contour[:, 0, 0], axis=0),
                               np.mean(contour[:, 0, 1], axis=0)]
        properties['orientation'] = 0
        properties['area'] = 0
        properties['perimeter'] = 0
        properties['aspect_ratio'] = 0
        return properties

    (_, (w_tilt, h_tilt), _) = cv.minAreaRect(contour)
    properties['center'] = [M['m10']/M['m00'],M['m01']/M['m00']]
    properties['orientation'] = (1/2)*np.arctan2(2*M['mu11'],(M['mu20']-M['mu02']))
    properties['area'] = M['m00']
    properties['perimeter'] = cv.arcLength(contour,True)
    properties['aspect_ratio'] = max(w_tilt, h_tilt) / min(w_tilt, h_tilt)
    return properties

def get_domain_properties(layers,shape):
    """
    Takes a 3D domain and returns properties
    """
    properties = {}
    properties['volume'] = 0
    properties['volume_dilated'] = 0
    properties['surface_area'] = 0

    weights = []
    center_xs = []
    center_ys = []
    center_zs = []
    orient_xs = []
    orient_ys = []

    for layer, domain_z in layers.items():
        center_zs.append(layer)
        if len(domain_z.keys()) == 1:
            subdomain = next(iter(domain_z.values()))
            properties['volume'] += subdomain['properties']['area']
            properties['volume_dilated'] += subdomain['properties']['area_dilated']
            properties['surface_area'] += subdomain['properties']['perimeter']

            weights.append(subdomain['properties']['area'])
            center_xs.append(subdomain['properties']['center'][0])
            center_ys.append(subdomain['properties']['center'][1])
            double_angles_rad = 2 * subdomain['properties']['orientation']
            x = np.cos(double_angles_rad)
            y = np.sin(double_angles_rad)
            orient_xs.append(x)
            orient_ys.append(y)

        else:
            masses = []
            xs = []
            ys = []
            for subdomain in domain_z.values():
                properties['volume'] += subdomain['properties']['area']
                properties['volume_dilated'] += subdomain['properties']['area_dilated']
                properties['surface_area'] += subdomain['properties']['perimeter']
                masses.append(subdomain['properties']['area'])
                xs.append(subdomain['properties']['center'][0])
                ys.append(subdomain['properties']['center'][1])

            centroid, orientation = compute_orientation(xs, ys, masses)
            weights.append(sum(masses))
            center_xs.append(centroid[0]%shape[0])
            center_xs.append(centroid[1]%shape[1])
            double_angles_rad = 2 * orientation
            x = np.cos(double_angles_rad)
            y = np.sin(double_angles_rad)
            orient_xs.append(x)
            orient_ys.append(y)

    if sum(weights) == 0:
        properties['center'] = np.array([np.mean(center_xs),np.mean(center_ys),np.mean(center_zs)])
        properties['orientation'] = 0
    else:
        # calculate center
        cx = (np.sum(unwrap_1D(center_xs,shape[0])*np.array(weights))/properties['volume']) % shape[0]
        cy = (np.sum(unwrap_1D(center_ys,shape[1])*np.array(weights))/properties['volume']) % shape[1]
        cz = np.sum(np.array(center_zs)*np.array(weights))/properties['volume']
        properties['center'] = np.array([cx,cy,cz])
        # calculate avg orientation
        orient_xs_avg = np.average(orient_xs,weights=weights)
        orient_ys_avg = np.average(orient_ys,weights=weights)
        properties['orientation'] = np.arctan2(orient_ys_avg, orient_xs_avg)/2
    return properties