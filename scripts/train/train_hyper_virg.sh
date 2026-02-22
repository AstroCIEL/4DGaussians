export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/vrig/broom2 --port 6068 --expname "hypernerf/vrig/broom2" --configs arguments/hypernerf/broom2.py &
export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/hypernerf/vrig/vrig-3dprinter --port 6066 --expname "hypernerf/vrig/vrig-3dprinter" --configs arguments/hypernerf/3dprinter.py &
export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/hypernerf/vrig/vrig-peel-banana --port 6069 --expname "hypernerf/vrig/vrig-peel-banana" --configs arguments/hypernerf/banana.py  &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/vrig/vrig-chicken --port 6070 --expname "hypernerf/vrig/vrig-chicken" --configs arguments/hypernerf/chicken.py 
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path output/hypernerf/broom2 --configs arguments/hypernerf/broom2.py --skip_train  --skip_test &
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path output/hypernerf/vrig-3dprinter  --configs arguments/hypernerf/3dprinter.py --skip_train  --skip_test &
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path output/hypernerf/vrig-peel-banana --configs arguments/hypernerf/banana.py --skip_train --skip_test &
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path output/hypernerf/vrig-chicken  --configs arguments/hypernerf/chicken.py --skip_train --skip_test &
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/hypernerf/broom2/"  &
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/hypernerf/3dprinter/" &
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/hypernerf/peel-banana/" &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/hypernerf/vrig-chicken/" &
wait
echo "Done"