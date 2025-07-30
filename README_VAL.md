CUDA_VISIBLE_DEVICES=1 python validation/validation_pipeline_landmarks.py --env test3_norm01_colorsrgb --num_query_points 85000 --top_k 4000 --num_points 4000 --model 4 --deciduous true --variable 2 --top_k_max 1200

rsync -avz -e "ssh -p 31415" grammatikakis1@dgxa100.icsd.hmu.gr:/home/grammatikakis1/p2-tree-gen/landmarks_austria/test3_norm01_colorsrgb/pointclouds-landmarks "/mnt/c/Users/mmddd/Documents/p2-tree-gen/landmarks_austria/models/test3_norm01_colorsrgb"

## 
cd validation
"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python convert_to_colored.py

python convert_to_splat.py

"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" --python gen_renderings.py

###
python gen_scenes.py
"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" --python gen_renderings_scene.py








CUDA_VISIBLE_DEVICES=5 python validation/validation_pipeline_landmarks.py --env mixed_ortho --num_query_points 50000 --top_k 4000 --num_points 4000 --model 6 --deciduous true --variable 2 --top_k_max 6000

rsync -avz -e "ssh -p 31415" grammatikakis1@dgxa100.icsd.hmu.gr:/home/grammatikakis1/p2-tree-gen/landmarks_austria/TREE_MODELS "/mnt/c/Users/mmddd/Documents/p2-tree-gen/landmarks_austria/"



"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python convert_to_pointy_trees.py (first part for Dense, second for Pointy)
"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python convert_to_colored.py
"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" --python gen_renderings.py