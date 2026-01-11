
python train.py -s data/dnerf/jumpingjacks --port 7169 --expname "dnerf/jumpingjacks" --configs arguments/dnerf/jumpingjacks.py  &
python train.py -s data/dnerf/trex --port 7170 --expname "dnerf/trex" --configs arguments/dnerf/trex.py 
# jumpingjacks ~20fps
python render.py --model_path "output/dnerf/jumpingjacks/"  --skip_train --configs arguments/dnerf/jumpingjacks.py
# trex ~15fps ~2GB
python render.py --model_path "output/dnerf/trex/"  --skip_train --configs arguments/dnerf/trex.py  
# 
python metrics.py --model_path "output/dnerf/jumpingjacks/"
python metrics.py --model_path "output/dnerf/trex/" 
wait
python train.py -s data/dnerf/mutant --port 7168 --expname "dnerf/mutant" --configs arguments/dnerf/mutant.py &
python train.py -s data/dnerf/standup --port 7166 --expname "dnerf/standup" --configs arguments/dnerf/standup.py 
# mutant ~20fps ~2GB
python render.py --model_path "output/dnerf/mutant/"  --skip_train --configs arguments/dnerf/mutant.py
# standup ~20fps
python render.py --model_path "output/dnerf/standup/"  --skip_train --configs arguments/dnerf/standup.py 
wait
python metrics.py --model_path "output/dnerf/mutant/" 
python metrics.py --model_path "output/dnerf/standup/"  
wait
python train.py -s data/dnerf/hook --port 7369 --expname "dnerf/hook" --configs arguments/dnerf/hook.py  &
python train.py -s data/dnerf/hellwarrior --port 7370 --expname "dnerf/hellwarrior" --configs arguments/dnerf/hellwarrior.py 
# hellwarrior ~20fps
python render.py --model_path "output/dnerf/hellwarrior/"  --skip_train --configs arguments/dnerf/hellwarrior.py
# hook ~20fps
python render.py --model_path "output/dnerf/hook/"  --skip_train --configs arguments/dnerf/hook.py  
wait
python metrics.py --model_path "output/dnerf/hellwarrior/"  &
python metrics.py --model_path "output/dnerf/hook/" 
wait
python train.py -s data/dnerf/lego --port 7168 --expname "dnerf/lego" --configs arguments/dnerf/lego.py &
python train.py -s data/dnerf/bouncingballs --port 7166 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
# bouncingballs ~18fps
python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py
# lego ~14fps
python render.py --model_path "output/dnerf/lego/"  --skip_train --configs arguments/dnerf/lego.py  
wait
python metrics.py --model_path "output/dnerf/bouncingballs/" &
python metrics.py --model_path "output/dnerf/lego/"   
wait
echo "Done"