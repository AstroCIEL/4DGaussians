python train.py -s data/hypernerf/virg/broom2 --port 6068 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py &
python train.py -s data/hypernerf/virg/vrig-3dprinter --port 6066 --expname "hypernerf/3dprinter" --configs arguments/hypernerf/3dprinter.py &
python train.py -s data/hypernerf/virg/peel-banana --port 6069 --expname "hypernerf/peel-banana" --configs arguments/hypernerf/banana.py  &
python train.py -s data/hypernerf/virg/vrig-chicken --port 6070 --expname "hypernerf/vrig-chicken" --configs arguments/hypernerf/chicken.py 
wait
python render.py --model_path output/hypernerf/vrig/broom2 --configs arguments/hypernerf/broom2.py --skip_train  --skip_video 
python render.py --model_path output/hypernerf/vrig/vrig-3dprinter  --configs arguments/hypernerf/vrig-3dprinter.py --skip_train  --skip_video 
python render.py --model_path output/hypernerf/vrig/vrig-peel-banana --configs arguments/hypernerf/vrig-peel-banana.py --skip_train --skip_video 
python render.py --model_path output/hypernerf/vrig/vrig-chicken  --configs arguments/hypernerf/vrig-chicken.py --skip_train --skip_video 
wait
python metrics.py --model_path "output/hypernerf/broom2/"  &
python metrics.py --model_path "output/hypernerf/3dprinter/" &
python metrics.py --model_path "output/hypernerf/peel-banana/" &
python metrics.py --model_path "output/hypernerf/vrig-chicken/" &
wait
echo "Done"