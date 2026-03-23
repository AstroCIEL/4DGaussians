
# 获取脚本所在目录，并切换到项目根目录
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
# cd "$PROJECT_ROOT" || exit 1

python simulator/main.py --config simulator/configs/dynerf/gscore_dynerf.yaml
python simulator/main.py --config simulator/configs/dynerf/gscore_dynerf_hilbert.yaml
python simulator/main.py --config simulator/configs/dynerf/gscore_dynerf_hilbert_fifo.yaml
python simulator/main.py --config simulator/configs/dynerf/gscore_dynerf_greedy.yaml

python simulator/main.py --config simulator/configs/hypernerf/gscore_hypernerf.yaml
python simulator/main.py --config simulator/configs/hypernerf/gscore_hypernerf_hilbert.yaml
python simulator/main.py --config simulator/configs/hypernerf/gscore_hypernerf_hilbert_fifo.yaml
python simulator/main.py --config simulator/configs/hypernerf/gscore_hypernerf_greedy.yaml
