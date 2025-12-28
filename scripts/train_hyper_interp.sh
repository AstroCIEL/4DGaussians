export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/hypernerf/interp/aleks-teapot --port 6568 --expname "hypernerf/interp/aleks-teapot" --configs arguments/hypernerf/default.py &
export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/hypernerf/interp/slice-banana --port 6566 --expname "hypernerf/interp/slice-banana" --configs arguments/hypernerf/default.py &
export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/hypernerf/interp/chickchicken --port 6569 --expname "hypernerf/interp/interp-chicken" --configs arguments/hypernerf/default.py & 

wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/interp/cut-lemon1 --port 6670 --expname hypernerf/interp/cut-lemon1 --configs arguments/hypernerf/default.py &
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/interp/hand1-dense-v2 --port 6671 --expname hypernerf/interp/hand1-dense-v2 --configs arguments/hypernerf/default.py &
export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/hypernerf/interp/torchocolate --port 6672 --expname hypernerf/interp/torchocolate --configs arguments/hypernerf/default.py &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path output/hypernerf/interp/aleks-teapot --configs arguments/hypernerf/default.py --skip_train &
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path output/hypernerf/interp/slice-banana  --configs arguments/hypernerf/default.py --skip_train &
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path output/hypernerf/interp/interp-chicken --configs arguments/hypernerf/default.py --skip_train &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path output/hypernerf/interp/cut-lemon1  --configs arguments/hypernerf/default.py --skip_train &
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path output/hypernerf/interp/hand1-dense-v2  --configs arguments/hypernerf/default.py --skip_train&
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path output/hypernerf/interp/torchocolate --configs arguments/hypernerf/default.py --skip_train &

wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/hypernerf/interp/aleks-teapot/"  &
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/hypernerf/interp/slice-banana/" &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/hypernerf/interp/interp-chicken/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/hypernerf/interp/cut-lemon1/" &
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/hypernerf/interp/hand1-dense-v2/" &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/hypernerf/interp/torchocolate/" 
wait
echo "Done"