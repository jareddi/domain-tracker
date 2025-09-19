from .tracker_utils import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv

pv.set_jupyter_backend('trame')

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
  Plotting Functions
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""
def angle_to_rgb(angles,mask=None):
    """
    Convert a 3D array of angles in radians (-π to π) to an RGB image
    using HSV color mapping (hue = angle).
    """
    # Normalize angles to [0, 1] for hue (HSV)
    hue = (angles + np.pi) / (2 * np.pi)  # map from [-π, π] to [0, 1]
    
    # Create HSV array: hue varies, saturation and value fixed at 1
    hsv = np.zeros(angles.shape + (3,))
    hsv[..., 0] = hue           # Hue ([0,1])
    hsv[..., 1] = 1.0           # Saturation
    hsv[..., 2] = 1.0           # Value (brightness)

    # Convert to RGB (mcolors or cv2 conversion)
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb = (rgb * 255).astype(np.uint8)

    if mask is not None:
        rgb[~mask] = [0, 0, 0]

    return rgb

def angle_to_hsv_colorwheel(res=512):
    """
    Generates an HSV colorwheel image where hue corresponds to angle from -π to π.
    """
    radius = res // 2
    y, x = np.ogrid[-radius:radius, -radius:radius]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)  # flip y to match image coordinates
    mask = r <= radius

    hsv = np.zeros((res, res, 3), dtype=np.float32)
    hsv[..., 0] = (theta + np.pi) / (2 * np.pi)  # hue: 0 to 1
    hsv[..., 1] = 1.0
    hsv[..., 2] = 1.0
    hsv[~mask] = 0  # outside circle set to black

    rgb = mcolors.hsv_to_rgb(hsv)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb

def plot_domains_z(domains_z, image_shape, color='lime',properties=True):
    plt.imshow(np.zeros(image_shape), cmap='gray')  # blank canvas
    for key in domains_z.keys():
        """
        print(f"Domain {key}")
        print(f"Center: {domains_z[key]['properties']['center']}")
        print(f"Area: {domains_z[key]['properties']['area']:.2f}")
        print(f"Perimeter: {domains_z[key]['properties']['perimeter']:.2f}")
        print(f"Aspect Ratio: {domains_z[key]['properties']['aspect_ratio']:.2f}")
        print(f"Orientation: {np.rad2deg(domains_z[key]['properties']['orientation']):.2f}°")
        """
        for contour in domains_z[key]['contours']:
            contour = contour.squeeze()
            if contour.ndim == 2:  # safeguard for contours with more than one point
                plt.plot(contour[:, 0], contour[:, 1], color=color)
        if properties:
            plt.text(domains_z[key]['properties']['center'][0]%image_shape[0],domains_z[key]['properties']['center'][1]%image_shape[1], f"{key}", fontsize=12, color='white', ha='center', va='center')
    plt.gca().invert_yaxis()  # OpenCV uses (0,0) top-left; matplotlib uses bottom-left
    plt.title(f"Contours")

def pyvista_draw_frame(plotter,domains,gamma,label_domains=True,font='arial',fontsize=10):
    theme = pv.themes.DocumentProTheme()
    theme.font.family = font
    theme.font.size = fontsize
    plotter.theme = theme

    mask = np.zeros(domains[0].shape).astype(bool)
    bps = []
    coms = []
    domain_id = []
    for key in domains.keys():
        if isinstance(key,int):
            mask |= domains[key].unpack_mask()
            bps += domains[key].bps
            coms.append(domains[key].properties['center'])
            domain_id.append(str(domains[key].global_id))
    coms = np.array(coms)
    coms[:,0] = coms[:,0] % domains[0].shape[0]
    coms[:,1] = coms[:,1] % domains[0].shape[1]

    # Convert to float for scalar field
    mask = np.transpose(mask,(2,0,1))
    scalars = mask.astype(float)

    # create flattened angles array
    #angles_to_plot = theta
    angles_to_plot = gamma
    angles = np.transpose(angles_to_plot,(2,1,0))

    # create rgb colors array from angles
    rgb = angle_to_rgb(angles)  # Should return (nx, ny, nz, 3)
    rgb_flat = rgb.reshape(-1, 3)

    # Create PyVista grid (ImageData expects (X, Y, Z) shape)
    grid = pv.ImageData(dimensions=scalars.shape[::-1],spacing=(1,1,1),origin=(0,0,0))
    grid.point_data["mask"] = scalars.flatten()
    grid.point_data["angles"] = angles.flatten()
    grid.point_data["colors"] = rgb_flat

    # Now you can contour it
    contour = grid.contour(isosurfaces=[0.5], scalars="mask")
    contour = contour.interpolate(grid, sharpness=1.0)

    # Dummy scalar range to show full colormap
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cloud = pv.PolyData(points)
    cloud["angles"] = np.array([-np.pi, np.pi])

    # Plot the contour surface
    scalar_bar_args = {
        'title': 'Domain Wall Helicity',
        'position_x': 0.05,  # From 0 (left) to 1 (right)
        'position_y': 0.05,  # From 0 (bottom) to 1 (top)
        'vertical': False,   # Set to horizontal for typical bottom layout
        'width': 0.3,        # Optional: bar width
        'height': 0.05,      # Optional: bar height
    }
    plotter.add_mesh(cloud, scalars="angles", cmap='hsv', point_size=0.0001, opacity=0, lighting=False, reset_camera=False,
                     show_scalar_bar=True, scalar_bar_args=scalar_bar_args)
    plotter.add_mesh(contour, scalars="colors", rgb=True, opacity=1, lighting=False, reset_camera=False) # use for actual colors
    if label_domains:
        plotter.add_point_labels(
            coms,
            domain_id,
            font_size=24,
            point_color=None,      # No point marker
            text_color='black',
            shape=None,            # No shape background
            always_visible=True    # Optional: keeps label visible even if occluded
        )
    # Add a small black sphere at each Bloch point
    radius = 1
    if bps:
        centers = np.array([bp[0] for bp in bps], dtype=np.float32)
        point_cloud = pv.PolyData(centers)
        glyphs = point_cloud.glyph(scale=False, geom=pv.Sphere(radius=radius), orient=False)
        plotter.add_mesh(glyphs, color='black', show_edges=False, lighting=False, reset_camera=False)

    # Add title text
    plotter.add_text("Domain Shell (Mz = 0) Helicities", position='upper_edge', font=font, font_size=fontsize)

def spherical_camera_pos(radius, elevation_deg, azimuth_deg, focal_point):
    elevation_rad = np.radians(elevation_deg)
    azimuth_rad = np.radians(azimuth_deg)

    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = radius * np.sin(elevation_rad)

    pos = np.array([x, y, z]) + focal_point
    return pos.tolist(), focal_point.tolist(), [0, 0, 1]  # camera_position tuple

def pyvista_generate_movie(frames, filename, framerate, radius=640, elevation=30, azimuth=-60, window_size=(720,720)):
    plotter = pv.Plotter(window_size=window_size, off_screen=False)
    plotter.open_movie(filename, framerate=framerate)

    # Precompute camera position
    camera = pv.Camera()
    camera.enable_parallel_projection()
    shape = np.array(frames[0]['domains'][0].shape, dtype=np.float32)
    radius = np.sqrt(np.sum(shape**2))
    focal_point = np.array(shape) / 2
    camera.position, camera.focal_point, camera.up = spherical_camera_pos(radius, elevation, azimuth, focal_point)
    camera.parallel_scale = max(shape)/2
    camera.clipping_range = (0, 3000)
    plotter.camera = camera

    steps_to_go = len(frames)
    for i, frame in enumerate(frames.values()):
        print(f"Progress: {100 * i // steps_to_go:.0f}%")
        plotter.clear()  # removes previous frame's actors
        pyvista_draw_frame(plotter, frame['domains'], frame['gamma'])
        #plotter.camera = camera
        plotter.write_frame()
    print(f"Progress: 100%")

    plotter.close()


