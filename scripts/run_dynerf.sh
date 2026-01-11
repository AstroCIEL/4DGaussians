python render.py --model_path output/dynerf/cut_roasted_beef --configs arguments/dynerf/cut_roasted_beef.py --skip_train 
python render.py --model_path output/dynerf/sear_steak --configs arguments/dynerf/sear_steak.py --skip_train 
python render.py --model_path output/dynerf/flame_steak --configs arguments/dynerf/flame_steak.py --skip_train 
python render.py --model_path output/dynerf/flame_salmon_1 --configs arguments/dynerf/flame_salmon_1.py --skip_train 
python render.py --model_path output/dynerf/cook_spinach  --configs arguments/dynerf/cook_spinach.py --skip_train  
python render.py --model_path output/dynerf/coffee_martini --configs arguments/dynerf/coffee_martini.py --skip_train 
python metrics.py --model_path "output/dynerf/cut_roasted_beef/"  
python metrics.py --model_path "output/dynerf/cook_spinach/" 
python metrics.py --model_path "output/dynerf/sear_steak/" 
python metrics.py --model_path "output/dynerf/flame_salmon_1/"  
python metrics.py --model_path "output/dynerf/flame_steak/" 
python metrics.py --model_path "output/dynerf/coffee_martini/" 
echo "Done"