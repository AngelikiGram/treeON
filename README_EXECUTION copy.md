ssh -L 8097:localhost:8097 -L 8098:localhost:8098 -L 8099:localhost:8099 -L 8090:localhost:8090 -L 8091:localhost:8091 -L 8092:localhost:8092 -p 31415 grammatikakis1@dgxa100.icsd.hmu.gr
source ~/.bashrc
cd network-tree-gen 

source ~/.bashrc
python -m visdom.server -port 8099

####

nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_all --bce true --shadow true --silhouettes true --model 1 --classes_loss true --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=4 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_all_noCl --bce true --shadow true --silhouettes true --model 1 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=5 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 13000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_all_noCl_dsm --bce true --shadow true --silhouettes true --model 1 --model_previous_training true --nepoch 650 --variable 2 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=1 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_all_noCl_colors --colors true --bce true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=2 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_all_Cl_colors --colors true --bce true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true --classes_loss true > out.log 2>&1` &


## COLORS! (gt_colors = histogram_match X)

nohup `CUDA_VISIBLE_DEVICES=6 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env noCl_colors --colors true --bce true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=7 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env Cl_colors --colors true --bce true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true --classes_loss true > out.log 2>&1` &


# from beginning no classes - colors
nohup `CUDA_VISIBLE_DEVICES=5 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env test --colors true --bce true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &



nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 2500 --num_query 25000 --num_trees 2500 --top_k 2500 --thres 25 --env all_2500 --bce true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &


## ABLATION (no Classes Loss)

nohup `CUDA_VISIBLE_DEVICES=0 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_bce_shadow --bce true --shadow true --model 1 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=5 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_bce_silhouettes --bce true --silhouettes true --model 1 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=6 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_bce --bce true --model 1 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=7 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_shadow_silhouettes --shadow true --silhouettes true --model 1 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &



rsync -avz -e "ssh -p 31415" --exclude='.git' --exclude='landmarks_austria/outputs' /mnt/c/Users/mmddd/Documents/network-tree-gen/ grammatikakis1@dgxa100.icsd.hmu.gr:~/network-tree-gen/

## VALIDATION 
CUDA_VISIBLE_DEVICES=5 python validation/validation_pipeline.py --env Cl_colors --num_query_points 50000 --top_k 3500 --num_points 3500 --model 5 --deciduous true --variable 3 --top_k_max 3500



# -------------------------------------
# -------------------------------------

CUDA_VISIBLE_DEVICES=0 python validation/validation_pipeline_landmarks.py --env mixed_all_noCl --num_query_points 50000 --top_k 2500 --num_points 2500 --model 1 --deciduous true --variable 3 --top_k_max 2500

rsync -avz -e "ssh -p 31415" grammatikakis1@dgxa100.icsd.hmu.gr:/home/grammatikakis1/network-tree-gen/landmarks_austria/TREE_MODELS "/mnt/c/Users/mmddd/Documents/network-tree-gen/landmarks_austria/"

cd validation
# Change model_name = "mixed_all_noCl" (XXX if colored)
"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --python convert_to_colored.py

# Saved in landmarks_austria/TREE_MODELS/COLORED/{model_name}

# ----
# Open Power Shell on supersample-pointcloud 
# Change config ({model_name})
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\batch-process.ps1 -InputFolder "C:\Users\mmddd\Documents\network-tree-gen\landmarks_austria\TREE_MODELS\COLORED\mixed_all_noCl"
# ----

# Change model_name + pointcloud_supersampled
"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" --background --python gen_renderings.py

# Saved in landmarks_austria/outputs/{model_name}

# -------------------------------------
# -------------------------------------

# FOR COLORS: 
CUDA_VISIBLE_DEVICES=6 python validation/validation_pipeline_landmarks.py --env mixed_all_noCl_colors --num_query_points 50000 --top_k 2500 --num_points 2500 --model 5 --deciduous true --variable 3 --top_k_max 2500









# ** CHECKS ** 
- have the colors
--> to check if the classes is better than without 
--> many-trees or not also to check 

nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1500 --num_query 20000 --num_trees 250 --top_k 1500 --thres 25 --env mixed_all_topk --bce
true --shadow true --silhouettes true --model 1 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &


-> train also without the rgb resnet in image (for better shadows)?










#### 
# test 
nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed --colors true --bce true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

##
<!-- NO REFINEMENT -->
nohup `CUDA_VISIBLE_DEVICES=5 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env noRefinement --colors true --bce true --shadow true --silhouettes true --model 10 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

## 

nohup `CUDA_VISIBLE_DEVICES=4 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_bce --colors true --bce true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=6 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_bce_shadow --colors true --bce true --shadow true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=7 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_bce_silhouettes --colors true --bce true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_silhouettes --colors true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=4 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_shadow --colors true --shadow true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=5 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_shadow_silhouettes --colors true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &



nohup `CUDA_VISIBLE_DEVICES=6 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env dsm_all --colors true --bce true --shadow true --silhouettes true --model 2 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=7 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env dsm_bce_shadow --colors true --bce true --shadow true --model 2 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &


nohup `CUDA_VISIBLE_DEVICES=2 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env dsm_shadow_silhouettes --colors true --shadow true --silhouettes true --model 2 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env dsm_shadow --colors true --shadow true true --model 2 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &


## 

nohup `CUDA_VISIBLE_DEVICES=0 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env dsm_bce_silhouettes --colors true --bce true --silhouettes true --model 2 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=1 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env dsm_bce --colors true --bce true --model 2 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=4 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env dsm_silhouettes --colors true --silhouettes true --model 2 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &



nohup `CUDA_VISIBLE_DEVICES=5 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_all --colors true --bce true --shadow true --silhouettes true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=6 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_silhouettes --colors true --bce true --silhouettes true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=7 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_shadow --colors true --bce true --shadow true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &



nohup `CUDA_VISIBLE_DEVICES=2 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_shadow_silhouettes_nobce --colors true --shadow true --silhouettes true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_shadow_nobce --colors true --shadow true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &


### RUN

nohup `CUDA_VISIBLE_DEVICES=1 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_all --colors true --bce true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=2 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_all_noRefinement --colors true --bce true --shadow true --silhouettes true --model 10 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &


nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_bce_shadow --colors true --bce true --shadow true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=4 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_bce_silhouettes --colors true --bce true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=5 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_shadow_silhouettes --colors true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=6 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_bce --colors true --bce true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=7 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_shadow --colors true --shadow true --model 5 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &




nohup `CUDA_VISIBLE_DEVICES=1 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env mixed_all --colors true --bce true --shadow true --silhouettes true --model 5 --model_previous_training true --nepoch 1000 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &



nohup `CUDA_VISIBLE_DEVICES=1 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_all --colors true --bce true --shadow true --silhouettes true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=2 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_silhouettes --colors true --silhouettes true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=3 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_bce_shadow --colors true --bce true --shadow true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=4 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_bce_silhouettes --colors true --bce true --silhouettes true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=5 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_shadow_silhouettes --colors true --shadow true --silhouettes true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=6 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_bce --colors true --bce true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &

nohup `CUDA_VISIBLE_DEVICES=7 python train1.py --port 8099 --image_size 90 --batchSize 16 --num_points 1250 --num_query 25000 --num_trees 1250 --top_k 1250 --thres 25 --env ortho_shadow --colors true --shadow true --model 3 --model_previous_training true --nepoch 650 --variable 3 --many_trees true --top_k_gt_occupancy true > out.log 2>&1` &




conda activate p2

nohup `python train/my_train.py --env svrtmnet_1_modified --use_shadow_loss True --use_silhouette_loss True > out.log 2>&1` &

nohup `python train/my_train.py --env svrtmnet_1_original --use_shadow_loss False --use_silhouette_loss False > out.log 2>&1` &



python validation/generate_output_pointcloud.py tree_0005 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/MINE/ --env mixed_all 
python validation/generate_output_pointcloud.py tree_0430 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/MINE/ --env mixed_all 
python validation/generate_output_pointcloud.py tree_0030 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/MINE/ --env mixed_all 

python validation/generate_output_pointcloud.py tree_0129 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/MINE/ --env mixed_all 
python validation/generate_output_pointcloud.py tree_0500 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/MINE/ --env mixed_all 
python validation/generate_output_pointcloud.py tree_0501 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/MINE/ --env mixed_all 
python validation/generate_output_pointcloud.py 1185 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/MINE/ --env mixed_all 
python validation/generate_output_pointcloud.py 1171 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/MINE/ --env mixed_all 



python validation/generate_output_pointcloud.py tree_0005 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/noRef/ --env mixed_all_noRefinement --num_points 600 --model 10
python validation/generate_output_pointcloud.py tree_0430 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/noRef/ --env mixed_all_noRefinement --model 10
python validation/generate_output_pointcloud.py tree_0030 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/noRef/ --env mixed_all_noRefinement --model 10

python validation/generate_output_pointcloud.py tree_0129 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/noRef/ --env mixed_all_noRefinement --model 10
python validation/generate_output_pointcloud.py tree_0500 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/noRef/ --env mixed_all_noRefinement --model 10
python validation/generate_output_pointcloud.py tree_0501 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/noRef/ --env mixed_all_noRefinement --model 10
python validation/generate_output_pointcloud.py 1185 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/noRef/ --env mixed_all_noRefinement --model 10

python validation/generate_output_pointcloud.py 1171 --dataset_root /home/grammatikakis1/TREES_DATASET --output_root /home/grammatikakis1/COMPARISONS_MODELS/noRef/ --env mixed_all_noRefinement --model 10