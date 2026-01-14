# --- Minimal tree renderer (Blender 4.x) -------------------------------------
# - PLY: renders colored point clouds via Geometry Nodes
# - OBJ: imports a mesh and (optionally) applies a histogram-matched texture
# ------------------------------------------------------------------------------

import sys, site, os, csv, math
site_path = site.getusersitepackages()
if site_path not in sys.path:
    sys.path.append(site_path)

import numpy as np
from skimage import io, exposure

import bpy
import bmesh
from mathutils import Vector, Euler

# ===================== CONFIG =====================

MODEL_NAME   = "mixed_bce_silhouettes" # TOCHANGE -- DSM
MODE_NAME = "colored" # TOCHANGE (colored, trunk, points)
supersample = False # True # TOCHANGE

dsm = False

# Color histogram matching config
ENABLE_HISTOGRAM_MATCHING = True # True  # Set to False to disable histogram matching for point clouds

# Camera placement config
CAM_DISTANCE = 85.0
CAM_RELATIVE_TO_SIZE = True  # <--- Set this boolean to choose camera placement mode
CAM_SIZE_FACTOR = 2.5        # Camera will be placed at (size * factor) from object center if CAM_RELATIVE_TO_SIZE is True

BASE = r"C:\Users\angie\Documents\treeON\results"
TREE_DIR     = os.path.join(BASE, "TREE_MODELS", MODEL_NAME, "pointclouds-landmarks") 

TREE_DIR = r"C:\Users\angie\Documents\treeON\landmarks_austria\outputs\temp"
if dsm:
    TREE_DIR     = r"C:\Users\angie\Documents\treeON\landmarks_austria\DATA_LANDMARKS\DSM_OBJ"
ORTHO_DIR    = os.path.join(BASE, "DATA_LANDMARKS", "ORTHOPHOTOS")
TEXTURE_DIR  = os.path.join(BASE, "textures")
TEMP_DIR     = os.path.join(BASE, "temp")
OUT_DIR      = os.path.join(TREE_DIR, "out") # os.path.join(BASE, "outputs", MODEL_NAME)
CSV_PATH     = os.path.join(BASE, "trees-data.csv")

## TOCHANGE: If not supersampled
if supersample: 
    TREE_DIR     = os.path.join("C:/Users/angie/Documents/P2/supersample_pointclouds/OUTPUTS_ablation/", MODE_NAME + "-" + MODEL_NAME) # + '1')   
    OUT_DIR      = os.path.join(BASE, "outputs", MODEL_NAME) # , f"supersample-{MODE_NAME}")

RENDER_SIZE  = (512, 512)
DEFAULT_POINT_RADIUS = 0.08 # 35 # 01 # fallback value
ROTATE_PC_X90 = False  # rotate point clouds around X by +90° if needed

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== HELPERS =====================
def load_categories(csv_path):
    m = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            m[row["ID"].zfill(3)] = row["Category"].strip().lower()
    return m

CATEGORY = load_categories(CSV_PATH)

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# def match_histograms(src_img, tgt_img, out_img):
#     src = io.imread(src_img).astype(float) / 255.0
#     tgt = io.imread(tgt_img).astype(float) / 255.0
#     matched = np.empty_like(src)
#     for c in range(3):
#         matched[..., c] = exposure.match_histograms(src[..., c], tgt[..., c])
#     io.imsave(out_img, (matched * 255).astype(np.uint8))

from skimage import exposure, io, img_as_float
import numpy as np

def match_histograms(src_img, tgt_img, out_img, ortho_emphasis=0.75):
    src = io.imread(src_img)
    tgt = io.imread(tgt_img)

    print(f"[DEBUG] src shape: {src.shape}, tgt shape: {tgt.shape}")

    # Resize target to match source if needed
    if src.shape[:2] != tgt.shape[:2]:
        from skimage.transform import resize
        tgt = resize(tgt, src.shape, preserve_range=True, anti_aliasing=True).astype(src.dtype)
        print(f"[DEBUG] Resized tgt to {tgt.shape}")

    # Stats before
    print(f"[DEBUG] src mean: {src.mean(axis=(0,1))}, std: {src.std(axis=(0,1))}")
    print(f"[DEBUG] tgt mean: {tgt.mean(axis=(0,1))}, std: {tgt.std(axis=(0,1))}")

    # Create output texture by randomly sampling pixels from both images
    h, w = src.shape[:2]
    output_texture = np.zeros_like(src, dtype=np.float32)
    
    # Normalize images
    src_normalized = src.astype(np.float32) / 255.0
    tgt_normalized = tgt.astype(np.float32) / 255.0
    
    # Random sampling approach
    np.random.seed(42)  # For reproducible results
    
    for y in range(h):
        for x in range(w):
            # Random decision based on ortho_emphasis probability
            if np.random.random() < ortho_emphasis:
                # Sample random pixel from orthophoto (target)
                rand_y = np.random.randint(0, tgt_normalized.shape[0])
                rand_x = np.random.randint(0, tgt_normalized.shape[1])
                output_texture[y, x] = tgt_normalized[rand_y, rand_x]
            else:
                # Sample random pixel from leaf texture (source)
                rand_y = np.random.randint(0, src_normalized.shape[0])
                rand_x = np.random.randint(0, src_normalized.shape[1])
                output_texture[y, x] = src_normalized[rand_y, rand_x]
    
    # Convert back to uint8
    emphasized_result = np.clip(output_texture * 255, 0, 255)

    # Stats after
    print(f"[DEBUG] output texture mean: {emphasized_result.mean(axis=(0,1))}, ortho emphasis: {ortho_emphasis}")
    print(f"[DEBUG] Random sampling: {ortho_emphasis*100:.1f}% ortho pixels, {(1-ortho_emphasis)*100:.1f}% leaf pixels")

    io.imsave(out_img, emphasized_result.astype(np.uint8))
    print(f"[DEBUG] Saved random-sampled texture to {out_img}")


def first_mesh():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    return meshes[0] if meshes else None

def setup_camera_and_light(target_obj, distance=CAM_DISTANCE):
    # Camera
    bbox = [target_obj.matrix_world @ Vector(c) for c in target_obj.bound_box]
    center = sum(bbox, Vector()) / 8.0
    # Compute bounding box size
    size = max((b - a).length for a, b in zip(bbox, bbox[1:] + bbox[:1]))
    if CAM_RELATIVE_TO_SIZE:
        cam_dist = size * CAM_SIZE_FACTOR
    else:
        cam_dist = distance
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(cam)
    cam.location = center + Vector((cam_dist, 0, 0))
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    # Light
    sun = bpy.data.lights.new(name="Sun", type='SUN')
    sun.energy = 3.0
    light_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, 5, 5)

def setup_render(path, size=RENDER_SIZE, transparent=True):
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.render.resolution_x, scn.render.resolution_y = size
    scn.render.filepath = path
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGBA'
    scn.render.film_transparent = transparent
    # World bg white
    world = scn.world or bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (1, 1, 1, 1)

# ---------- Point cloud loader + GN renderer ----------
def blend_point_colors_with_leaf_texture(point_colors, leaf_colors, point_weight=0.7, leaf_weight=0.3):
    """Blend point cloud colors with leaf texture colors using weighted combination"""
    if leaf_colors is None or len(leaf_colors) == 0:
        return point_colors
    
    blended_colors = []
    num_leaf_colors = len(leaf_colors)
    
    for i, (pr, pg, pb, pa) in enumerate(point_colors):
        # Select a leaf color (cycle through available colors)
        leaf_idx = i % num_leaf_colors
        lr, lg, lb = leaf_colors[leaf_idx]
        
        # Blend colors using weighted combination
        blended_r = point_weight * pr + leaf_weight * lr
        blended_g = point_weight * pg + leaf_weight * lg
        blended_b = point_weight * pb + leaf_weight * lb
        
        # Clamp values to [0,1] range
        blended_r = max(0.0, min(1.0, blended_r))
        blended_g = max(0.0, min(1.0, blended_g))
        blended_b = max(0.0, min(1.0, blended_b))
        
        blended_colors.append((blended_r, blended_g, blended_b, pa))
    
    return blended_colors

def load_points_with_colors(path):
    ext = os.path.splitext(path)[1].lower()
    verts, cols = [], []
    if ext == ".ply":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            if not f.readline().startswith("ply"):
                raise ValueError("Not a PLY file.")
            format_ascii = False
            nverts = None
            prop = []
            in_vert = False
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected EOF in PLY header.")
                s = line.strip()
                if s.startswith("format"):
                    format_ascii = "ascii" in s
                    if not format_ascii:
                        raise ValueError("Only ASCII PLY is supported.")
                elif s.startswith("element"):
                    parts = s.split()
                    in_vert = (len(parts) == 3 and parts[1] == "vertex")
                    if in_vert:
                        nverts = int(parts[2])
                elif s.startswith("property") and in_vert:
                    prop.append(s.split()[-1])
                elif s == "end_header":
                    break
            ix, iy, iz = prop.index("x"), prop.index("y"), prop.index("z")
            # Check if color properties exist
            has_colors = "red" in prop and "green" in prop and "blue" in prop
            if has_colors:
                ir, ig, ib = prop.index("red"), prop.index("green"), prop.index("blue")
            
            for _ in range(nverts):
                parts = f.readline().split()
                if len(parts) < len(prop): continue
                x, y, z = float(parts[ix]), float(parts[iy]), float(parts[iz])
                verts.append(Vector((x, y, z)))
                
                if has_colors:
                    r, g, b = int(parts[ir]), int(parts[ig]), int(parts[ib])
                    cols.append((r/255.0, g/255.0, b/255.0, 1.0))
                else:
                    # Default to white color if no color properties
                    cols.append((1.0, 1.0, 1.0, 1.0))
    elif ext == ".obj":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.strip().split()
                    if len(p) >= 7:
                        x, y, z = map(float, p[1:4])
                        r, g, b = map(float, p[4:7])
                        verts.append(Vector((x, y, z)))
                        cols.append((r, g, b, 1.0))
    else:
        raise ValueError(f"Unsupported point format: {ext}")
    if ROTATE_PC_X90:
        R = Euler((math.radians(90), 0, 0), 'XYZ').to_matrix().to_4x4()
        verts = [(R @ v.to_4d()).xyz for v in verts]
    return verts, cols

def create_gn_pointcloud(verts, colors, radius, lit=True):
    # make mesh with only vertices + POINT color attribute "Col"
    mesh = bpy.data.meshes.new("PointCloudMesh")
    obj  = bpy.data.objects.new("PointCloud", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.from_pydata([tuple(v) for v in verts], [], [])
    mesh.update()

    col_attr = mesh.color_attributes.new(name="Col", type='FLOAT_COLOR', domain='POINT')
    n = len(mesh.vertices)
    
    # Create a copy of colors to avoid modifying the original list
    colors_copy = list(colors)
    
    # Ensure colors list has exactly n elements
    if len(colors_copy) < n:
        # Pad with white color if not enough colors
        colors_copy = colors_copy + [(1, 1, 1, 1)] * (n - len(colors_copy))
        print(f"[DEBUG] Padded colors from {len(colors)} to {n} vertices")
    elif len(colors_copy) > n:
        # Truncate if too many colors
        colors_copy = colors_copy[:n]
        print(f"[DEBUG] Truncated colors from {len(colors)} to {n} vertices")
    
    # Safely assign colors
    for i in range(n):
        if i < len(colors_copy):
            col_attr.data[i].color = colors_copy[i]
        else:
            col_attr.data[i].color = (1, 1, 1, 1)  # Fallback white color
            print('[WARNING] Assigned fallback white color to vertex', i)

    # Geometry Nodes: Mesh->Points + Set Material
    mod = obj.modifiers.new(name="GN_Points", type='NODES')
    ng = bpy.data.node_groups.new("GN_PointCloud", 'GeometryNodeTree')
    mod.node_group = ng
    nodes, links = ng.nodes, ng.links
    nodes.clear()

    gin  = nodes.new('NodeGroupInput')
    gout = nodes.new('NodeGroupOutput')
    try:
        ng.interface.new_socket("Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
        ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    except TypeError:
        pass

    n_m2p  = nodes.new('GeometryNodeMeshToPoints')
    n_setm = nodes.new('GeometryNodeSetMaterial')

    # point size
    if "Radius" in n_m2p.inputs: n_m2p.inputs["Radius"].default_value = radius
    elif "Size" in n_m2p.inputs:  n_m2p.inputs["Size"].default_value   = radius

    # material that reads "Col"
    mat = bpy.data.materials.new("PointColMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for nd in list(nt.nodes):
        if nd.type != 'OUTPUT_MATERIAL': nt.nodes.remove(nd)
    out = nt.nodes["Material Output"]
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "Col"
    if lit:
        diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
        nt.links.new(attr.outputs["Color"], diff.inputs["Color"])
        nt.links.new(diff.outputs["BSDF"], out.inputs["Surface"])
    else:
        emit = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    n_setm.inputs['Material'].default_value = mat
    links.new(gin.outputs['Geometry'], n_m2p.inputs['Mesh'])
    links.new(n_m2p.outputs['Points'], n_setm.inputs['Geometry'])
    links.new(n_setm.outputs['Geometry'], gout.inputs['Geometry'])
    return obj

# ---------- Mesh OBJ importer (with optional matched texture) ----------
def import_mesh_obj(path, texture_img=None, dsm=False):
    print('PATH:', path)
    bpy.ops.wm.obj_import(filepath=path)
    obj = first_mesh()
    if not obj:
        raise RuntimeError("OBJ import failed.")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()

    if dsm:
        return obj

    if texture_img and os.path.exists(texture_img):
        img = bpy.data.images.load(texture_img)
        mat = bpy.data.materials.new("MeshTex"); mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        for n in list(nodes):
            if n.type != 'OUTPUT_MATERIAL': nodes.remove(n)
        out = nodes["Material Output"]
        tex = nodes.new("ShaderNodeTexImage"); tex.image = img
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        obj.data.materials.clear(); obj.data.materials.append(mat)
        # quick UV
        bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(); bpy.ops.object.mode_set(mode='OBJECT')
    return obj

def match_histogram_texture(source_img_file, target_img_file, output_file, blend_ratio=0.7):
    """Apply histogram matching only to non-transparent pixels with blending"""
    source_img = io.imread(source_img_file).astype(float) / 255.0
    target_img = io.imread(target_img_file).astype(float) / 255.0
    
    # Handle RGBA vs RGB
    if source_img.shape[2] == 4:  # RGBA
        rgb_source = source_img[:, :, :3]
        alpha_channel = source_img[:, :, 3]
        # Create mask for non-transparent pixels
        non_transparent_mask = alpha_channel > 0.01  # Small threshold for floating point
    else:  # RGB
        rgb_source = source_img
        non_transparent_mask = np.ones(source_img.shape[:2], dtype=bool)  # All pixels
    
    if target_img.shape[2] == 4:
        target_img = target_img[:, :, :3]  # Use only RGB from target
    
    # Apply histogram matching only to non-transparent pixels
    matched_rgb = np.copy(rgb_source)
    for c in range(3):
        if np.any(non_transparent_mask):
            # Extract non-transparent pixels for matching
            source_channel = rgb_source[:, :, c][non_transparent_mask]
            target_channel = target_img[:, :, c].ravel()
            
            # Match histograms
            matched_channel = exposure.match_histograms(source_channel, target_channel)
            
            # Put matched values back into the image (only for non-transparent pixels)
            matched_rgb[:, :, c][non_transparent_mask] = matched_channel
    
    # Blend original and matched (only for non-transparent pixels)
    blended_rgb = np.copy(rgb_source)
    blended_rgb[non_transparent_mask] = (
        blend_ratio * rgb_source[non_transparent_mask] + 
        (1.0 - blend_ratio) * matched_rgb[non_transparent_mask]
    )
    
    # Reconstruct final image
    if source_img.shape[2] == 4:  # Keep original alpha
        final_img = np.dstack([blended_rgb, alpha_channel])
    else:
        final_img = blended_rgb
    
    # Clip and save
    final_img = np.clip(final_img, 0.0, 1.0)
    io.imsave(output_file, (final_img * 255).astype(np.uint8))
    
    print(f"[DEBUG] Histogram matched {np.sum(non_transparent_mask)} non-transparent pixels with {blend_ratio*100:.1f}% original blend")

def match_point_colors(cols, ortho_png, blend_ratio=0.7, enable_histogram_matching=True):
    """Match per-point RGB colors to the histogram of the orthophoto image with spatial sampling."""
    if not enable_histogram_matching:
        print("[DEBUG] Histogram matching disabled, returning original colors.")
        return cols
        
    if not os.path.exists(ortho_png):
        print("[DEBUG] No ortho found, skipping histogram match.")
        return cols

    try:
        # Load orthophoto
        ortho = io.imread(ortho_png)
        if ortho.shape[-1] == 4:  # Drop alpha
            ortho = ortho[..., :3]
        elif ortho.ndim == 2:  # Grayscale -> RGB
            ortho = np.stack([ortho]*3, axis=-1)
        
        # Simple histogram matching fallback (original method)
        ortho = ortho.astype(np.float32) / 255.0
        pts = np.array([c[:3] for c in cols], dtype=np.float32)

        matched = np.zeros_like(pts)
        for ch in range(3):  # match each channel separately
            matched[:, ch] = exposure.match_histograms(
                pts[:, ch], ortho[..., ch].ravel()
            )

        matched = np.clip(matched, 0, 1)
        
        # Blend original and matched colors
        blended = blend_ratio * pts + (1.0 - blend_ratio) * matched
        blended = np.clip(blended, 0, 1)
        
        print(f"[DEBUG] Point colors mean before: {pts.mean(axis=0)}, after: {blended.mean(axis=0)} (blend ratio: {blend_ratio:.1f})")
        return [(r, g, b, 1.0) for r, g, b in blended]
    except Exception as e:
        print(f"[DEBUG] Histogram match for PLY failed: {e}")
        return cols

# ===================== MAIN =====================
for fname in sorted(os.listdir(TREE_DIR)):
    if not fname.lower().endswith((".ply", ".obj")):
        continue

    stem = os.path.splitext(fname)[0]
    try:
        tree_id = stem.split('_')[1]  # expects e.g., "tree_001_mesh"
    except IndexError:
        tree_id = ''.join(ch for ch in stem if ch.isdigit()).zfill(3)

    PARENT_DIR = os.path.dirname(OUT_DIR)
    out_png = os.path.join(PARENT_DIR, f"{MODEL_NAME}.png") # OUTDIR tree_id
    if os.path.exists(out_png):
        print(f"Skip {tree_id}: already rendered.")
      #  continue

    # if '31' not in tree_id:
    #     print(f"Skip {tree_id}: not in '6xx' series.")
    #     continue

    category = CATEGORY.get(tree_id, "deciduous")
    leaf_tex = os.path.join(TEXTURE_DIR, "coniferous.jpg" if category == "coniferous" else "deciduous.jpg")
    ortho_png = os.path.join(ORTHO_DIR, f"ortho_{tree_id}.png")

    clear_scene()

    path = os.path.join(TREE_DIR, fname)
    ext  = os.path.splitext(path)[1].lower()

    # Save leaf texture colors to text file
    def extract_and_save_leaf_colors(texture_path, tree_id, output_dir):
        """Extract colors from leaf texture and save to text file"""
        try:
            img = io.imread(texture_path)
            if img.shape[-1] == 4:  # Remove alpha if present
                img = img[..., :3]
            
            # Get representative colors by random sampling
            h, w = img.shape[:2]
            sample_points = 100  # Number of color samples
            
            # Random sampling instead of uniform grid
            np.random.seed(hash(tree_id) % 2**32)  # Reproducible per tree_id
            colors = []
            for i in range(sample_points):
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                r, g, b = img[y, x]
                colors.append([r/255.0, g/255.0, b/255.0])  # Normalize to [0,1]
            
            print(f"Extracted {len(colors)} random leaf colors from {texture_path}")
            return colors
            
        except Exception as e:
            print(f"Failed to extract leaf colors: {e}")
            return None

    # Try histogram matching leaf texture to the ortho (if present)
    matched_tex = os.path.join(TEMP_DIR, f"{tree_id}_leaf_matched.jpg")
    try:        
        print("Leaf texture: ", leaf_tex, " Ortho: ", ortho_png, " Matched: ", matched_tex)
        if os.path.exists(ortho_png):
            match_histograms(leaf_tex, ortho_png, matched_tex, ortho_emphasis=0.45)
            tex_to_use = matched_tex
        else:
            tex_to_use = leaf_tex
    except Exception as e:
        print(f"Histogram match failed for {tree_id}: {e}")
        tex_to_use = leaf_tex

    leaf_tex = leaf_tex.replace('deciduous', 'coniferous')
    
    # Extract leaf colors and save to text file
    leaf_colors = extract_and_save_leaf_colors(tex_to_use, tree_id, OUT_DIR)

    if ext == ".ply":
        verts, cols = load_points_with_colors(path)
        if not verts:
            print(f"⚠️ No points in {fname}; skipping.")
            continue
        # Compute bounding box size for point cloud
        coords = np.array([v.to_tuple() for v in verts])
        min_xyz = coords.min(axis=0)
        max_xyz = coords.max(axis=0)
        tree_size = np.linalg.norm(max_xyz - min_xyz)
        point_radius = DEFAULT_POINT_RADIUS # max(tree_size * 0.005, 0.029)  # 1% of size, min fallback  * 0.0005 0.01
        print('point radius:', point_radius)
        # obj = create_gn_pointcloud(verts, cols, radius=point_radius, lit=True)

        # Blend point cloud colors with leaf texture colors
        if leaf_colors:
            original_cols = cols.copy()  # Keep original for comparison
            cols = blend_point_colors_with_leaf_texture(cols, leaf_colors, point_weight=0.85, leaf_weight=0.15)
            # Final Color = (0.6 × Point Cloud Color) + (0.4 × Leaf Texture Color)
            print(f"Blended {len(cols)} point colors with leaf texture (60% point, 40% leaf)")
        
        # Create point cloud without histogram matching (will be done on final image)
        obj = create_gn_pointcloud(verts, cols, radius=point_radius, lit=True)

    else:  # ".obj" mesh
        obj = import_mesh_obj(path, texture_img=tex_to_use, dsm=dsm)
        # Compute bounding box size for mesh
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_xyz = np.min([v.to_tuple() for v in bbox], axis=0)
        max_xyz = np.max([v.to_tuple() for v in bbox], axis=0)
        tree_size = np.linalg.norm(np.array(max_xyz) - np.array(min_xyz))
        point_radius = max(tree_size * 0.01, 0.002)  # 1% of size, min fallback
        # If you want to use point_radius for mesh rendering, pass it to relevant functions here

    obj.location = (0, 0, 0)
    obj.scale    = (1, 1, 1)

    setup_camera_and_light(obj)
    setup_render(out_png, size=RENDER_SIZE, transparent=True)
    bpy.ops.render.render(write_still=True)
   
    
    print(f"✅ Rendered {tree_id} -> {out_png}")

# Example CLI:
# "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python gen_renderings_min.py
