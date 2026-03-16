export CUDA_VISIBLE_DEVICES=3

# 获取脚本所在目录，并切换到项目根目录
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
# cd "$PROJECT_ROOT" || exit 1

python simulator/main.py --config simulator/configs/default_dynerf.yaml
python simulator/main.py --config simulator/configs/gscore_dynerf.yaml
python simulator/main.py --config simulator/configs/neo_dynerf.yaml
# python simulator/main.py --config simulator/configs/default_hypernerf.yaml
# python simulator/main.py --config simulator/configs/gscore_hypernerf.yaml
# python simulator/main.py --config simulator/configs/neo_hypernerf.yaml