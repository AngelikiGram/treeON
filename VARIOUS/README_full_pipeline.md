# 1. Server connect
ssh -L 8097:localhost:8097 -L 8098:localhost:8098 -L 8099:localhost:8099 -L 8090:localhost:8090 -L 8091:localhost:8091 -L 8092:localhost:8092 -p 31415 grammatikakis1@dgxa100.icsd.hmu.gr
source ~/.bashrc
cd network-tree-gen 

source ~/.bashrc
python -m visdom.server -port 8099

# 2. Train model (--classes_loss true)
nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_all --bce true --shadow true --silhouettes true --model 5 --classes_loss true --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

# 3.1 Update the code
rsync -avz -e "ssh -p 31415" --exclude='.git' --exclude='landmarks_austria/outputs' --exclude='landmarks_austria/TREE_MODELS' /mnt/c/Users/mmddd/Documents/P2/network-tree-gen/ grammatikakis1@dgxa100.icsd.hmu.gr:~/network-tree-gen/
# 3.2 Download Tree Models 
rsync -avz -e "ssh -p 31415" grammatikakis1@dgxa100.icsd.hmu.gr:/home/grammatikakis1/network-tree-gen/landmarks_austria/TREE_MODELS "/mnt/c/Users/mmddd/Documents/P2/network-tree-gen/landmarks_austria/"

# 4. Validation - ablation 
CUDA_VISIBLE_DEVICES=5 python validation/validation_pipeline.py --env test --num_query_points 50000 --top_k 2500 --num_points 2500 --model 5 --deciduous true --variable 3 --top_k_max 2500

# -------------------------------------
# -------------------------------------
# 5. Validation - qualitative
CUDA_VISIBLE_DEVICES=0 python validation/validation_pipeline_landmarks.py --env test --num_query_points 50000 --top_k 2500 --num_points 2500 --model 1 --deciduous true --variable 3 --top_k_max 2500

rsync -avz -e "ssh -p 31415" grammatikakis1@dgxa100.icsd.hmu.gr:/home/grammatikakis1/network-tree-gen/landmarks_austria/TREE_MODELS "/mnt/c/Users/mmddd/Documents/P2/network-tree-gen/landmarks_austria/"

# --> locally: 

# ---- supersample-pointcloud ----
# Open Power Shell on supersample-pointcloud 
# TOCHANGE a) config ({model_name} & {mode_name}), b) command
# TOCHANGE: If + trunk: c) trunk-creation.py ()
blender --python _VARIOUS/trunk-creation.py

Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\batch-process.ps1 -InputFolder "C:\Users\mmddd\Documents\P2\network-tree-gen\landmarks_austria\TREE_MODELS\COLORED\mixed_all_noCl"
# ----

# ---- network-tree-gen ----
cd validation
# TOCHANGE MODEL_NAME & MODE_NAME & supersample
"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --background --python gen_renderings.py

# Saved in landmarks_austria/outputs/{model_name}

# -------------------------------------
# -------------------------------------





## If no colors:
cd validation
# Change model_name = "mixed_all_noCl" (XXX if colored)
"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python convert_to_colored.py




# Scenes Generation 
cd validation/Scenes_Gen 
python gen_scenes.py
"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python gen_renderings_scene.py


##
8, 18, 19
12

23
39
25
52
51
55
70
3
5
6
