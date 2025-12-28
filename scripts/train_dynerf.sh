export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dynerf/flame_salmon_1 --port 6468 --expname "dynerf/flame_salmon_1" --configs arguments/dynerf/flame_salmon_1.py &
export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/dynerf/coffee_martini --port 6469 --expname "dynerf/coffee_martini" --configs arguments/dynerf/coffee_martini.py  &
wait
export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dynerf/cook_spinach --port 6436 --expname "dynerf/cook_spinach" --configs arguments/dynerf/cook_spinach.py &
# wait
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/dynerf/cut_roasted_beef --port 6470 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py 
wait 
export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dynerf/flame_steak      --port 6471 --expname "dynerf/flame_steak" --configs arguments/dynerf/flame_steak.py &
export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/dynerf/sear_steak       --port 6569 --expname "dynerf/sear_steak" --configs arguments/dynerf/sear_steak.py  
wait

export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path output/dynerf/cut_roasted_beef --configs arguments/dynerf/cut_roasted_beef.py --skip_train &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path output/dynerf/sear_steak --configs arguments/dynerf/sear_steak.py --skip_train 
wait
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path output/dynerf/flame_steak --configs arguments/dynerf/flame_steak.py --skip_train &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path output/dynerf/flame_salmon_1 --configs arguments/dynerf/flame_salmon_1.py --skip_train 
wait
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path output/dynerf/cook_spinach  --configs arguments/dynerf/cook_spinach.py --skip_train  &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path output/dynerf/coffee_martini --configs arguments/dynerf/coffee_martini.py --skip_train &
wait
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/dynerf/cut_roasted_beef/"  &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/dynerf/cook_spinach/" 
wait
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/dynerf/sear_steak/" &
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/dynerf/flame_salmon_1/"  
wait
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/dynerf/flame_steak/" &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/dynerf/coffee_martini/" 
echo "Done"