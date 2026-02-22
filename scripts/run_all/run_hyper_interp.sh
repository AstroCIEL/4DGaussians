python render.py --model_path output/hypernerf/interp/aleks-teapot --configs arguments/hypernerf/default.py --skip_train
python render.py --model_path output/hypernerf/interp/slice-banana  --configs arguments/hypernerf/default.py --skip_train
python render.py --model_path output/hypernerf/interp/interp-chicken --configs arguments/hypernerf/default.py --skip_train
python render.py --model_path output/hypernerf/interp/cut-lemon1  --configs arguments/hypernerf/default.py --skip_train
python render.py --model_path output/hypernerf/interp/hand1-dense-v2  --configs arguments/hypernerf/default.py --skip_train
python render.py --model_path output/hypernerf/interp/torchocolate --configs arguments/hypernerf/default.py --skip_train

wait
python metrics.py --model_path "output/hypernerf/interp/aleks-teapot/"  &
python metrics.py --model_path "output/hypernerf/interp/slice-banana/" &
python metrics.py --model_path "output/hypernerf/interp/interp-chicken/" 
python metrics.py --model_path "output/hypernerf/interp/cut-lemon1/" &
python metrics.py --model_path "output/hypernerf/interp/hand1-dense-v2/" &
python metrics.py --model_path "output/hypernerf/interp/torchocolate/" 
wait
echo "Done"