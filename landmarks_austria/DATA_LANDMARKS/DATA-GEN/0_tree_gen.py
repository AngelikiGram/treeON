import bpy
import os
import sys
import numpy as np
import random
from mathutils import Vector
import bmesh
import shutil
import time

def clear_output_folder(out_dir):
    """Clear the output folder."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

def clear_scene():
    """Clear the current Blender scene."""
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

def BarkMaterialPlace(texture_path, material_name):
    """Assign a bark texture to the tree object."""
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(name=material_name)
    obj = bpy.context.active_object
    obj.data.materials.append(material)

    material.use_nodes = True
    nodes = material.node_tree.nodes
    shader_node = nodes.get("Principled BSDF")

    image_texture_node = nodes.new(type="ShaderNodeTexImage")
    image_texture_node.location = (-200, 0)
    image_texture = bpy.data.images.load(texture_path)
    image_texture_node.image = image_texture

    material.node_tree.links.new(image_texture_node.outputs["Color"], shader_node.inputs["Base Color"])

def attach_twigs(tree_obj, twigs_directory, species):
    """Attach twig models to the tree."""
    twig_name = f"{species.lower()}.obj"
    twig_path = os.path.join(twigs_directory, twig_name)
    if not os.path.exists(twig_path):
        print(f"Twig model not found for species: {species}")
        return

    # Import twig model
    # bpy.ops.import_scene.obj(filepath=twig_path)
    bpy.ops.wm.obj_import(filepath=twig_path)
    twig_obj = bpy.context.selected_objects[0]
    twig_obj.name = "Twig_Model"

    # Attach twigs to tree branches
    bpy.context.view_layer.objects.active = tree_obj
    tree_obj.select_set(True)
    twig_obj.select_set(True)

    # Parent twig to tree
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

def get_object_by_substring(substring):
    for obj in bpy.data.objects:
        if substring in obj.name and 'Conifer' not in obj.name:
            return obj
    return None

def createTreeModel(species, size, bark_textures_directory, twigs_directory):
    """Create a tree model with bark and twigs."""
    clear_scene()
    addon_name = "the_grove_21_in_blender"
    if addon_name not in bpy.context.preferences.addons.keys():
        bpy.ops.preferences.addon_enable(module=addon_name)

    # Generate tree using The Grove add-on
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.the_grove_21.add('INVOKE_DEFAULT')
    properties = bpy.data.collections["Grove"].GROVE21_Properties
    properties.simulation_flushes = 1
    properties.presets_menu = species

    # Simulate tree growth
    properties.auto_prune_low = 2
    for _ in range(25):
        bpy.ops.the_grove_21.grow('INVOKE_DEFAULT', do_regrow=False)
        
    tree_obj = bpy.context.active_object

    # Scale tree to desired size
    tree_obj.scale = (size / tree_obj.dimensions.z,) * 3

    # Apply bark texture
    simplified_species = species
    if 'Spruce' in species or 'spruce' in species:
        simplified_species = 'Spruce'
    elif 'Fir' in species or 'fir' in species:
        simplified_species = 'Spruce'
    elif 'Pine' in species or 'pine' in species:
        simplified_species = 'Pine'
    elif 'Birch' in species or 'birch' in species:
        simplified_species = 'Birch'
    elif 'Poplar' in species or 'poplar' in species:
        simplified_species = 'Poplar'
    elif 'Cherry' in species or 'cherry' in species:
        simplified_species = 'Cherry'
    elif 'Chestnut' in species or 'chestnut' in species:
        simplified_species = 'Chestnut'
    elif 'Maple' in species or 'maple' in species:
        simplified_species = 'Maple'
    elif 'Hornbeam' in species or 'hornbeam' in species:
        simplified_species = 'Hornbeam'
    elif 'Linden' in species or 'linden' in species:
        simplified_species = 'Linden'
    elif 'Oak' in species or 'oak' in species:
        simplified_species = 'Oak'
    elif 'Plane' in species or 'plane' in species:
        simplified_species = 'Plane'
    elif 'Willow' in species or 'willow' in species:
        simplified_species = 'Willow'
    elif 'Elm' in species or 'elm' in species:
        simplified_species = 'Elm'
    elif 'Ash' in species or 'ash' in species:
        simplified_species = 'Ash'
    elif 'Alder' in species or 'alder' in species:
        simplified_species = 'Alder'
    elif 'Beech' in species or 'beech' in species:
        simplified_species = 'Beech'
    elif 'Apple' in species or 'apple' in species:
        simplified_species = 'Apple'
    elif 'Aspen' in species or 'aspen' in species:
        simplified_species = 'Aspen'
    
    # Extract the texture filename without the path or extension
    for file in os.listdir(bark_textures_directory):
        if simplified_species.lower() in file.lower() and 'Normal' not in file:
            print(f"Found bark folder: {simplified_species.lower()}")
            matching_texture = os.path.join(bark_textures_directory, file)
            if matching_texture in bpy.context.collection.GROVE21_Properties.bl_rna.properties['texture_bark'].enum_items.keys():
                bpy.context.collection.GROVE21_Properties.texture_bark = matching_texture
            else:
                print(f"Warning: {matching_texture} is not a valid enum value for texture_bark.")
            break
    else:
        print(f"Warning: No matching bark texture found for {species}.")
    
    material_name = "TheGroveBranches"
    BarkMaterialPlace(bpy.context.collection.GROVE21_Properties.texture_bark, material_name)

    bpy.ops.object.select_all(action='DESELECT') 
    twig_name = f"{simplified_species.lower()}.obj"
    twig_filepath = os.path.join(twigs_directory, twig_name)
    bpy.ops.wm.obj_import(filepath=twig_filepath)
    imported_object = get_object_by_substring(simplified_species)
    bpy.context.view_layer.objects.active = imported_object
    imported_object.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    if imported_object:
        grove_twigs_material = "TheGroveTwigs"
        for folder in os.listdir(twigs_directory):
            if os.path.isdir(os.path.join(twigs_directory, folder)) and simplified_species.lower() in folder.lower():
                print(f"Found twigs folder: {simplified_species.lower()}")
                folder_twigs = folder
        texture_twigs = os.path.join(twigs_directory, folder_twigs, f"{simplified_species.lower()}.png")
        BarkMaterialPlace(texture_twigs, grove_twigs_material)

        bpy.data.collections["Grove"].GROVE21_Properties.twig_object_side = imported_object
        bpy.data.collections["Grove"].GROVE21_Properties.twig_object_upward = imported_object
        bpy.data.collections["Grove"].GROVE21_Properties.twig_object_end = imported_object
        bpy.data.collections["Grove"].GROVE21_Properties.twig_object_dead = imported_object

        # bpy.data.collections["Grove"].GROVE21_Properties.twig_density = 0.2

        bpy.data.collections["Grove"].GROVE21_Properties.twig_menu = 'Scene Objects'

    #     imported_object.select_set(True)
    #     bpy.context.view_layer.objects.active = imported_object
    #     bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    #     height = tree_obj.dimensions.z
    #     imported_object.location.z = height
    #     bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    return tree_obj

def optimize_tree_processing(tree_obj):
    """
    Optimize tree processing by reducing vertices, skipping unnecessary operations, 
    and batching updates.
    Args:
        tree_obj: The tree object to process.
    """
    # Ensure we are in Object Mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Select the tree object by name
    if tree_obj is None:
        print("Object 'Tree' not found.")
    else:
        bpy.context.view_layer.objects.active = tree_obj
        tree_obj.select_set(True)

        bpy.context.view_layer.update()

        # Ensure Decimate is applied first
        bpy.context.view_layer.update()

def optimize_tree_processing1(tree_obj):
    """
    Optimize tree processing by skipping unnecessary operations and batching updates.
    Args:
        tree_obj: The tree object to process.
    """
    # Ensure we are in Object Mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Select the tree object by name
    if tree_obj is None:
        print("Object 'Tree' not found.")
    else:
        bpy.context.view_layer.objects.active = tree_obj
        tree_obj.select_set(True)

        bpy.context.view_layer.update()
        # bpy.ops.object.duplicates_make_real()

        ## 
        for obj in tree_obj.children: 
            obj.select_set(True)
        bpy.ops.object.duplicates_make_real(use_base_parent=True, use_hierarchy=True)

        bpy.context.view_layer.update()

        if bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        tree_obj.select_set(True)
        bpy.context.view_layer.objects.active = tree_obj
        bpy.ops.object.join()

def save_tree_as_ply(tree_obj, out_dir, filename):
    """
    Save the tree object as a .ply file.
    Args:
        tree_obj: The tree object to save.
        out_dir: Output directory.
        filename: Name of the file.
    """
    start_time = time.time()

    # Ensure the tree is selected
    bpy.ops.object.select_all(action='DESELECT')
    tree_obj.select_set(True)
    bpy.context.view_layer.objects.active = tree_obj

    obj_file_path = os.path.join(out_dir, f"{filename}.obj")
    bpy.ops.wm.obj_export(filepath=obj_file_path)
    print(f"Exported tree and snow to: {obj_file_path}")

def normalize_species_probabilities(species_array):
    # Extract the probabilities from the input array
    probabilities = [item[1] for item in species_array]
    
    # Normalize the probabilities to sum up to 1
    total = sum(probabilities)
    normalized_species_array = [(item[0], item[1] / total) for item in species_array]
    
    return normalized_species_array


def apply_remesh_modifier(input_file, output_file):
    # Clear existing objects
    # bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import the .obj file
    # bpy.ops.import_scene.obj(filepath=input_file)
    print(f"Importing {input_file}...")
    bpy.ops.wm.obj_import(filepath=input_file)

    # take the first object in the scene
    obj = bpy.context.view_layer.objects[0]
    obj.select_set(True)
    
    # Add Remesh modifier
    remesh_modifier = obj.modifiers.new(name='Remesh', type='REMESH')
    remesh_modifier.mode = 'SMOOTH'
    remesh_modifier.octree_depth = 6  # Adjust as needed
    remesh_modifier.use_remove_disconnected = False

    # Apply the modifier
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier="Remesh")
     
    # Export the modified object
    # bpy.ops.export_scene.obj(filepath=output_file)
    bpy.ops.wm.obj_export(filepath=output_file)

def main():
    species_array = [('Pinaceae - Austrian pine', 0.1), ('Pinaceae - Lodgepole pine', 0.25), ('Pinaceae - Longleaf pine', 0.2), ('Pinaceae - Ponderosa pine', 0.2), 
                     ('Pinaceae - Scots pine', 0.2), ('Pinaceae - Stone pine', 0.05)]

    # species_array = [
    #     ('Betulaceae - Alder', 0.1), ('Rosaceae - Wild apple', 0.02), ('Oleaceae - Ash', 0.1), ('Salicaceae - Aspen', 0.02), ('Fagaceae - Beech', 0.2),
    #     ('Betulaceae - Silver birch', 0.1), 
    #     ('Rosaceae - Japanese cherry', 0.1), ('Ulmaceae - Elm', 0.03), ('Salicaceae - Grey poplar', 0.08), ('Salicaceae - Italian poplar', 0.02), ('Betulaceae - Hornbeam', 0.04), ('Sapindaceae - Horse chestnut', 0.03), ('Malvaceae - Linden', 0.1),
    #     ('Sapindaceae - Maple', 0.04), ('Sapindaceae - Field maple', 0.06), ('Fagaceae - European oak', 0.1), ('Platanaceae - London plane tree', 0.1), ('Salicaceae - Willow', 0.03)]
    # species_array = [    
    #     ('Conifer - Spruce', 0.15), ('Conifer - Norway Spruce', 0.15), 
    #     ('Conifer - Silver Fir', 0.1), ('Conifer - Siberian Spruce', 0.05)
    # ] # ('Conifer - Scots Pine', 0.15), ('Conifer - Austrian Pine', 0.05)
    # Normalize probabilities
    normalized_species_array = normalize_species_probabilities(species_array)
    
    # Display normalized species probabilities
    print("Normalized Species Probabilities:")
    for species, prob in normalized_species_array:
        print(f"{species}: {prob:.4f}")
    
    # Example of random choice based on probabilities
    species, probabilities = zip(*normalized_species_array)
    chosen_species = random.choices(species, weights=probabilities, k=10)
    print("/nRandomly Chosen Species:")
    print(chosen_species)


    # Normalize probabilities to ensure they sum to 1
    species, probabilities = zip(*species_array)
    probabilities = np.array(probabilities) / sum(probabilities)

    # Output directory
    # out_dir = os.path.join(os.getcwd(), "conifers") if "--out" not in sys.argv else os.path.join(os.getcwd(), sys.argv[sys.argv.index("--out") + 1])
    # out_dir = os.path.join(os.getcwd(), "deciduous") if "--out" not in sys.argv else os.path.join(os.getcwd(), sys.argv[sys.argv.index("--out") + 1]) #c
    out_dir = "F:/TREES/NEW"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    # clear_output_folder(out_dir)
    bark_textures_directory = os.path.join("C:/Users/mmddd/Documents/the_grove_21/templates", "BarkTextures")
    twigs_directory = os.path.join("C:/Users/mmddd/Documents/the_grove_21/templates", "TwigsLibrary")

    train_dir = os.path.join(out_dir, "train")
    test_dir = os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    num_train = 1000 
    total_trees = num_train #+ num_test

    # 9, 237
    for i in range(1, total_trees + 1):
        counter = i + 2000
        if i <= num_train:
            ply_dir = train_dir
        else:
            ply_dir = test_dir

        ply_dir = test_dir

        output_file_path = os.path.join(ply_dir, f"tree_{counter:04d}.obj")
        if os.path.exists(output_file_path.replace('G:/', 'F:/')):
            print(f"File {output_file_path} already exists. Skipping...")
            continue

        # Select a species based on the defined probabilities
        species_choice = np.random.choice(species, p=probabilities)
        size = random.uniform(5, 50)
        tree_obj = createTreeModel(species_choice, size, bark_textures_directory, twigs_directory)

        ## apply_remesh_modifier
        ## decimate

        optimize_tree_processing(tree_obj)
        save_tree_as_ply(tree_obj, ply_dir, f"tree_{counter:04d}") # {species_choice.replace(' ', '')}")

if __name__ == "__main__":
    main()