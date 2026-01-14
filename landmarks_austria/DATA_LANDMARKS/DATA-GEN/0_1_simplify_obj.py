# import pymeshlab as ml
import os

def clear_scene():
    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Delete all meshes, materials, images, etc.
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block, do_unlink=True)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block, do_unlink=True)
    for block in bpy.data.images:
        bpy.data.images.remove(block, do_unlink=True)
    for block in bpy.data.textures:
        bpy.data.textures.remove(block, do_unlink=True)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block, do_unlink=True)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block, do_unlink=True)

# import sys
# import site

# # Manually add the user site-packages directory to sys.path
# site_path = site.getusersitepackages()
# if site_path not in sys.path:
#     sys.path.append(site_path)
# import pymeshlab as ml
# def simplify_obj(input_obj, output_obj, target_vertex_count):
#     import os
#     import pymeshlab as ml

#     ms = ml.MeshSet()
#     ms.load_new_mesh(input_obj)

#     print(f"Original mesh: {ms.current_mesh().vertex_number()} vertices, {ms.current_mesh().face_number()} faces")

#     # Apply uniform Poisson-disk sampling for point cloud simplification
#     ms.apply_filter(
#         'generate_sampling_poisson_disk',
#         samplenum=target_vertex_count,
#         exactnumflag=True,
#         bestsampleflag=True,
#         bestsamplepool=10,
#         exactnumtolerance=0.005
#     )

#     simplified_vertices = ms.current_mesh().vertex_number()
#     print(f"Simplified point cloud: {simplified_vertices} vertices")

#     # Save as OBJ (without faces, vertex-only if possible)
#     ms.save_current_mesh(output_obj, save_vertex_color=True, save_face_color=False, save_wedge_texcoord=False)

#     print(f"Saved simplified point cloud as OBJ: {output_obj}")

# def process_folder(input_folder, output_folder, target_vertex_count):
#     """
#     Processes all .obj files in the input folder, simplifies them, and saves them in the output folder.
    
#     Args:
#         input_folder (str): Path to the folder containing the input .obj files.
#         output_folder (str): Path to the folder to save the simplified .obj files.
#         target_vertex_count (int): Desired number of vertices in the simplified meshes.
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for file_name in os.listdir(input_folder):
#         if file_name.endswith('.obj'):
#             # check from tree_0110.obj and onwards
#             if int(file_name.split('_')[1].split('.')[0]) < 110:
#                 continue
#             input_file = os.path.join(input_folder, file_name)
#             output_file = os.path.join(output_folder, file_name)
#             print(f"Processing {input_file}...")
#             simplify_obj(input_file, output_file, target_vertex_count)

# # Example usage
# input_folder = 'F:/TREES/NEW/test' # train'
# output_folder = 'F:/TREES/NEW/less_vertices'
# os.makedirs(output_folder, exist_ok=True)
# target_vertex_count = 5000 # 3000 # 2500

# process_folder(input_folder, output_folder, target_vertex_count)


# # python 0_1_simplify_obj.py

# # "C:\Program Files\Blender Foundation\Blender 4.3\4.3\python\bin\python.exe" -m pip install pymeshlab
# # import sys
# # import site

# # # Manually add the user site-packages directory to sys.path
# # site_path = site.getusersitepackages()
# # if site_path not in sys.path:
# #     sys.path.append(site_path)

## ---------------------------------------------------------------------------------------------------------------

import bpy
import os

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def apply_voxel_remesh(obj, voxel_size=0.05):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    remesh = obj.modifiers.new(name='Remesh', type='REMESH')
    remesh.mode = 'VOXEL'
    remesh.voxel_size = voxel_size
    remesh.use_remove_disconnected = False

    bpy.ops.object.modifier_apply(modifier='Remesh')

from mathutils import Vector
def process_obj(input_file, output_file):
    clear_scene()
    bpy.ops.wm.obj_import(filepath=input_file)
    pointcloud_obj = bpy.context.view_layer.objects[0]

    # # Parameters
    # Get Z height of the object
    bpy.context.view_layer.update()  # Ensure bounds are up to date
    bbox = [pointcloud_obj.matrix_world @ Vector(corner) for corner in pointcloud_obj.bound_box]
    z_values = [v.z for v in bbox]
    z_height = max(z_values) - min(z_values)
    # Set sphere radius proportional to z height
    sphere_radius = z_height * 0.02   # You can tweak this multiplier
    remesh_voxel_size = z_height * 0.01  # Also scaled accordingly
    print(f"Z Height = {z_height:.3f}, Sphere Radius = {sphere_radius:.3f}, Voxel Size = {remesh_voxel_size:.3f}")

    # # Ensure the object exists
    # pointcloud_obj = bpy.data.objects.get(pointcloud_obj_name)
    if not pointcloud_obj:
        raise ValueError(f"Object '{pointcloud_obj.name}' not found.")

    # Create Geometry Nodes modifier
    mod = pointcloud_obj.modifiers.new(name="GeoNodes", type='NODES')

    # Create new Geometry Node tree
    gn_tree = bpy.data.node_groups.new("PointCloudToMesh", 'GeometryNodeTree')
    mod.node_group = gn_tree

    # Clear nodes
    nodes = gn_tree.nodes
    links = gn_tree.links
    nodes.clear()

    # Add input/output nodes
    input_node = nodes.new("NodeGroupInput")
    input_node.location = (-600, 0)
    output_node = nodes.new("NodeGroupOutput")
    output_node.location = (800, 0)

    # Define sockets on the group interface
    gn_tree.interface.new_socket(name="Geometry", socket_type='NodeSocketGeometry', in_out='INPUT')
    gn_tree.interface.new_socket(name="Geometry", socket_type='NodeSocketGeometry', in_out='OUTPUT')

    # Add Icosphere (used to generate mesh from points)
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.location = (-400, -200)
    ico.inputs["Radius"].default_value = sphere_radius
    ico.inputs["Subdivisions"].default_value = 2

    # Instance on Points
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.location = (-200, 0)

    # Realize Instances
    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.location = (0, 0)

    # Mesh to Volume
    m2v = nodes.new("GeometryNodeMeshToVolume")
    m2v.location = (200, 0)
    # Set Voxel Size (usually input index 2)
    m2v.inputs[2].default_value = remesh_voxel_size

    # Volume to Mesh
    v2m = nodes.new("GeometryNodeVolumeToMesh")
    v2m.location = (400, 0)
    # Set Voxel Size (usually input index 1)
    v2m.inputs[1].default_value = remesh_voxel_size

    # Connect the nodes
    links.new(input_node.outputs[0], instance.inputs["Points"])
    links.new(ico.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], m2v.inputs[0])  # Mesh input
    links.new(m2v.outputs[0], v2m.inputs[0])                # Volume input
    links.new(v2m.outputs[0], output_node.inputs[0])        # Final mesh output

    bpy.ops.wm.obj_export(filepath=output_file)
    print(f"Exported: {output_file}")

def process_folder1(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for f in os.listdir(input_folder):
        if not f.endswith('.obj'):
            continue
        input_file = os.path.join(input_folder, f)
        output_file = os.path.join(output_folder, f)
        process_obj(input_file, output_file)

        # return

# === USAGE ===
input_folder = 'F:/TREES/NEW/less_vertices'
output_folder = 'F:/TREES/NEW/simplified_voxel'

process_folder1(input_folder, output_folder)

# blender --python 0_1_simplify_obj.py