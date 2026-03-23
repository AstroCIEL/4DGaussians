
# 获取脚本所在目录，并切换到项目根目录
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
# cd "$PROJECT_ROOT" || exit 1
python simulator/main.py --config simulator/configs/dse_k/dynerf_default_fifo.yaml
python simulator/main.py --config simulator/configs/dse_k/dynerf_global_greedy.yaml
python simulator/main.py --config simulator/configs/dse_k/dynerf_hilbert_4.yaml
python simulator/main.py --config simulator/configs/dse_k/dynerf_hilbert_8.yaml
python simulator/main.py --config simulator/configs/dse_k/dynerf_hilbert_16.yaml
python simulator/main.py --config simulator/configs/dse_k/dynerf_hilbert_32.yaml

python simulator/main.py --config simulator/configs/dse_k/hypernerf_default_fifo.yaml
python simulator/main.py --config simulator/configs/dse_k/hypernerf_global_greedy.yaml
python simulator/main.py --config simulator/configs/dse_k/hypernerf_hilbert_4.yaml
python simulator/main.py --config simulator/configs/dse_k/hypernerf_hilbert_8.yaml
python simulator/main.py --config simulator/configs/dse_k/hypernerf_hilbert_16.yaml
python simulator/main.py --config simulator/configs/dse_k/hypernerf_hilbert_32.yaml

