import bpy
import os

def get_lowest_middle_point(obj):
    verts = obj.data.vertices
    if not verts:
        return None
    
    # Find middle bounds in X and Y
    x_coords = [v.co.x for v in verts]
    y_coords = [v.co.y for v in verts]
    z_coords = [v.co.z for v in verts]
    
    x_mid = (min(x_coords) + max(x_coords)) / 2
    y_mid = (min(y_coords) + max(y_coords)) / 2
    
    # Find vertices close to the middle in X and Y
    tolerance = (max(x_coords) - min(x_coords)) * 0.1  # 10% tolerance
    middle_verts = [v for v in verts 
                   if abs(v.co.x - x_mid) <= tolerance and abs(v.co.y - y_mid) <= tolerance]
    
    if not middle_verts:
        # Fallback: use all vertices
        middle_verts = verts
    
    # Find lowest point among middle vertices
    lowest = min(middle_verts, key=lambda v: v.co.z)
    return lowest.co

import random
import bmesh
import mathutils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def is_inside_bvh(bvh, point, max_hits=100):
    """Check if a point is inside a closed mesh using ray parity (odd hit count = inside)."""
    # Test with multiple ray directions to ensure accuracy
    directions = [
        Vector((1, 0, 0)),
        Vector((0, 1, 0)), 
        Vector((0, 0, 1)),
        Vector((-1, 0, 0))
    ]
    
    inside_votes = 0
    
    for direction in directions:
        hits = 0
        origin = point.copy()
        
        for _ in range(max_hits):
            location, normal, face_index, distance = bvh.ray_cast(origin, direction)
            if location is None:
                break
            hits += 1
            # Move past the hit point to continue ray casting
            origin = location + direction * 1e-4
            
        # Odd number of hits means inside
        if hits % 2 == 1:
            inside_votes += 1
    
    # Point is inside if majority of rays indicate inside
    return inside_votes >= len(directions) // 2 + 1

def create_points_inside_mesh(obj, max_points=10000, attempts_multiplier=20):
    # Convert to evaluated depsgraph object to get modifiers applied if any
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    # Create BVH tree for inside-mesh checks
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    # Don't recalculate normals - use mesh as-is
    bvh = mathutils.bvhtree.BVHTree.FromBMesh(bm)

    # Get a tighter bounding box by using actual mesh vertices
    world_verts = [obj.matrix_world @ v.co for v in mesh.vertices]
    min_corner = Vector((min(v[i] for v in world_verts) for i in range(3)))
    max_corner = Vector((max(v[i] for v in world_verts) for i in range(3)))
    
    # Add small padding to avoid edge cases
    padding = 0.01
    min_corner -= Vector((padding, padding, padding))
    max_corner += Vector((padding, padding, padding))

    # Random sampling with improved inside detection
    points = []
    attempts = 0
    max_attempts = max_points * attempts_multiplier
    consecutive_failures = 0
    max_consecutive_failures = 1000

    print(f"Generating points inside mesh '{obj.name}' with {len(mesh.vertices)} vertices...")

    while len(points) < max_points and attempts < max_attempts and consecutive_failures < max_consecutive_failures:
        x = random.uniform(min_corner.x, max_corner.x)
        y = random.uniform(min_corner.y, max_corner.y)
        z = random.uniform(min_corner.z, max_corner.z)
        pt = Vector((x, y, z))

        # Check if point is inside the mesh using improved ray casting
        if is_inside_bvh(bvh, pt):
            points.append(pt)
            consecutive_failures = 0
            if len(points) % 500 == 0:
                print(f"Generated {len(points)} points so far...")
        else:
            consecutive_failures += 1

        attempts += 1

    # Create point cloud as mesh
    if points:
        mesh_data = bpy.data.meshes.new("PointCloud")
        point_obj = bpy.data.objects.new("PointCloud", mesh_data)
        bpy.context.collection.objects.link(point_obj)
        mesh_data.from_pydata(points, [], [])
        mesh_data.update()

        print(f"Successfully created {len(points)} points inside mesh '{obj.name}' after {attempts} attempts")
    else:
        print(f"Warning: Could not generate any points inside mesh '{obj.name}' after {attempts} attempts")
        point_obj = None

    # Cleanup
    bm.free()
    eval_obj.to_mesh_clear()

    return point_obj


def get_lowest_middle_point_combined(mesh_obj, point_obj=None):
    """Get the lowest middle point specifically from the generated points."""
    if not point_obj or not point_obj.data.vertices:
        print("No generated points found!")
        return None
    
    # Get only the generated points in world coordinates
    point_verts = [point_obj.matrix_world @ v.co for v in point_obj.data.vertices]
    print(f"Processing {len(point_verts)} generated points")
    
    # Find middle bounds in X and Y using only the generated points
    x_coords = [p.x for p in point_verts]
    y_coords = [p.y for p in point_verts]
    
    x_mid = (min(x_coords) + max(x_coords)) / 2
    y_mid = (min(y_coords) + max(y_coords)) / 2
    
    # Find points close to the middle in X and Y
    x_range = max(x_coords) - min(x_coords)
    tolerance = x_range * 0.1 if x_range > 0 else 0.1  # 10% tolerance with fallback
    
    middle_points = [p for p in point_verts 
                    if abs(p.x - x_mid) <= tolerance and abs(p.y - y_mid) <= tolerance]
    
    if not middle_points:
        # Fallback: use points near the geometric center
        center_tolerance = x_range * 0.2 if x_range > 0 else 0.2
        middle_points = [p for p in point_verts 
                        if abs(p.x - x_mid) <= center_tolerance and abs(p.y - y_mid) <= center_tolerance]
    
    if not middle_points:
        # Final fallback: use all generated points
        middle_points = point_verts
    
    # Find lowest point among middle points from generated points only
    lowest = min(middle_points, key=lambda p: p.z)
    print(f"Found lowest middle point from generated points at: {lowest} from {len(middle_points)} middle points out of {len(point_verts)} generated points")
    return lowest

# --- User config ---
TREES_DIR = "./DSM_OBJ"
DSM_DIR = "./DSM_OBJ"
OUTPUT_DIR = "./DSM_OBJ_NORMALIZED_ALIGNED"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tree_files = [f for f in os.listdir(TREES_DIR) if f.lower().endswith('.obj')]
dsm_files = [f for f in os.listdir(DSM_DIR) if f.lower().endswith('.obj')]

for tree_file in tree_files:
    tree_id = os.path.splitext(tree_file)[0].split('_')[-1]
    # If tree_id is 5 digits and starts with '0', remove leading zero
    if tree_id.isdigit() and len(tree_id) == 5 and tree_id.startswith('0'):
        tree_id_search = tree_id.lstrip('0')
    else:
        tree_id_search = tree_id
    # Try to find matching DSM file by tree_id or tree_id_search
    dsm_file = next((f for f in dsm_files if tree_id in f or tree_id_search in f), None)
    if not dsm_file:
        print(f"No DSM file found for {tree_file}")
        continue


    out_path = os.path.join(OUTPUT_DIR, f"tree_{tree_id}.obj")
    if os.path.exists(out_path):
        print(f"WARNING: Skipping {tree_file}, output already exists.")
      #  continue

    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    tree_path = os.path.join(TREES_DIR, tree_file)
    dsm_path = os.path.join(DSM_DIR, dsm_file)

    bpy.ops.wm.obj_import(filepath=tree_path, use_split_objects=False, use_split_groups=False)
    tree_obj = bpy.context.selected_objects[0]
    bpy.ops.wm.obj_import(filepath=dsm_path, use_split_objects=False, use_split_groups=False)
    dsm_obj = bpy.context.selected_objects[0]

    # apply all transforms to both objects
    bpy.context.view_layer.objects.active = tree_obj
    tree_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.view_layer.objects.active = dsm_obj
    dsm_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Move DSM object so its lowest middle point is at (0, 0, 0)
    bpy.context.view_layer.objects.active = dsm_obj
    
    # Generate points inside mesh and capture the returned point object
    point_obj = create_points_inside_mesh(dsm_obj, max_points=8000)

    print(f"DSM object aligned to origin and saved to {out_path}")

    # Get the lowest middle point from the generated points
    lowest_middle_point = get_lowest_middle_point_combined(dsm_obj, point_obj)
    if lowest_middle_point:
        # Calculate offset to move lowest middle point to origin
        offset = -lowest_middle_point
        print(f"Applying offset {offset} to both DSM and point cloud")
        
        # Apply offset to DSM object
        dsm_obj.location += offset
        bpy.context.view_layer.objects.active = dsm_obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # Apply the same offset to the point cloud
        if point_obj:
            point_obj.location += offset
            bpy.context.view_layer.objects.active = point_obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            print(f"Both DSM and point cloud moved by offset {offset} to position lowest middle point at origin")
        else:
            print(f"DSM moved by offset {offset} (no point cloud to move)")
    else:
        print("Could not find lowest middle point from generated points")

    # delete the tree object, keep only aligned DSM
    bpy.ops.object.select_all(action='DESELECT')
    tree_obj.select_set(True)
    bpy.ops.object.delete()

    bpy.ops.wm.obj_export(filepath=out_path)

    # if '51' in tree_file:
    #     break

# cd C:/Users/mmddd/Documents/P2/network-tree-gen/landmarks_austria/DATA_LANDMARKS
# "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" --background --python align_dsm.py