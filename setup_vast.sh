#!/usr/bin/env bash
# Setup script for Qwen3-TTS Italian fine-tuning on Vast.ai
# Can be executed directly with: curl -sSL <URL> | bash
# start with: wget --show-progress -q https://github.com/OhMySimo/Qwen3-TTS-finetuning/releases/download/startup/setup_vast.sh;

set -e

echo "=========================================="
echo "🚀 Qwen3-TTS Italian Training Setup"
echo "=========================================="
echo ""

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- AGGIUNTA: CONTROLLO E INSTALLAZIONE TMUX ---
echo "🔍 Checking for tmux..."
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}📦 tmux non trovato. Installazione in corso...${NC}"
    apt update && apt install tmux -y
    echo -e "${GREEN}✓${NC} tmux installato."
else
    echo -e "${GREEN}✓${NC} tmux è già presente."
fi

# Abilita il mouse in tmux per scorrere i log facilmente
echo "set -g mouse on" > ~/.tmux.conf
echo -e "${GREEN}✓${NC} Configurazione mouse per tmux applicata."
echo ""
# -----------------------------------------------

# Check if we're on Vast.ai (optional)
if [ ! -d "/workspace" ]; then
    echo -e "${YELLOW}⚠️  Warning: /workspace not found. Creating it...${NC}"
    mkdir -p /workspace
fi

cd /workspace

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ Error: nvidia-smi not found!${NC}"
    echo "This script requires a GPU instance."
    exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}✓${NC} Found $NUM_GPUS GPU(s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: python3 not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python $(python3 --version | cut -d' ' -f2)"

# Check pip
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ Error: pip not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} pip available"
echo ""

# Clone repository
echo "=========================================="
echo "📦 Cloning repository..."
echo "=========================================="
echo ""

if [ -d "Qwen3-TTS-finetuning" ]; then
    echo -e "${YELLOW}Repository already exists, updating...${NC}"
    cd Qwen3-TTS-finetuning
    git pull
    cd ..
else
    git clone https://github.com/OhMySimo/Qwen3-TTS-finetuning.git
fi

cd Qwen3-TTS-finetuning/finetuning
WORK_DIR=$(pwd)

echo -e "${GREEN}✓${NC} Repository ready at: $WORK_DIR"
echo ""

# Rinomina train_raw.jsonl esistente (se presente)
if [ -f "train_raw.jsonl" ]; then
    echo "📝 Renaming existing train_raw.jsonl to train_raw_correct.jsonl..."
    mv train_raw.jsonl train_raw_correct.jsonl
    echo -e "${GREEN}✓${NC} Renamed to train_raw_correct.jsonl"
fi

# Install dependencies
echo "=========================================="
echo "📚 Installing dependencies..."
echo "=========================================="
echo ""

echo "Installing core packages (this may take a few minutes)..."

# Determine pip command
PIP_CMD="pip"
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
fi

# Install with --break-system-packages for Vast.ai containers
$PIP_CMD install --break-system-packages --quiet qwen-tts
echo -e "${GREEN}✓${NC} qwen-tts"

$PIP_CMD install --break-system-packages --quiet accelerate
echo -e "${GREEN}✓${NC} accelerate"

$PIP_CMD install --break-system-packages --quiet tensorboard
echo -e "${GREEN}✓${NC} tensorboard"

$PIP_CMD install --break-system-packages --quiet soundfile
echo -e "${GREEN}✓${NC} soundfile"

echo ""
echo "Installing optional packages for better performance..."

# Flash Attention (optional but recommended)
if $PIP_CMD install --break-system-packages --quiet flash-attn --no-build-isolation 2>/dev/null; then
    echo -e "${GREEN}✓${NC} flash-attention (faster training)"
else
    echo -e "${YELLOW}⚠${NC} flash-attention (skipped, will use default attention)"
fi

# Bitsandbytes for 8-bit Adam (optional)
if $PIP_CMD install --break-system-packages --quiet bitsandbytes 2>/dev/null; then
    echo -e "${GREEN}✓${NC} bitsandbytes (8-bit optimizer)"
else
    echo -e "${YELLOW}⚠${NC} bitsandbytes (skipped, will use standard optimizer)"
fi

echo ""

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "import torch; print('  ✓ PyTorch ' + torch.__version__ + (' (CUDA)' if torch.cuda.is_available() else ' (CPU)'))" || exit 1
python3 -c "import qwen_tts; print('  ✓ qwen-tts')" || exit 1
python3 -c "import accelerate; print('  ✓ accelerate')" || exit 1
echo ""

# Download dataset
echo "=========================================="
echo "📊 Downloading Italian dataset..."
echo "=========================================="
echo ""

# Check if train_raw_correct.jsonl exists and is valid
if [ -f "train_raw_correct.jsonl" ]; then
    NUM_SAMPLES=$(wc -l < train_raw_correct.jsonl 2>/dev/null || echo "0")
    if [ "$NUM_SAMPLES" -gt "1000" ]; then
        echo -e "${GREEN}✓${NC} Dataset already exists ($NUM_SAMPLES samples)"
        SKIP_DOWNLOAD=1
    else
        echo -e "${YELLOW}⚠${NC} Dataset file exists but seems incomplete, re-downloading..."
        rm -f train_raw_correct.jsonl audio_dataset/ datasetv2.zip
        SKIP_DOWNLOAD=0
    fi
else
    SKIP_DOWNLOAD=0
fi

if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
    echo "Downloading dataset v2 (~500MB)..."
    
    if ! wget --show-progress -q https://github.com/OhMySimo/Qwen3-TTS-finetuning/releases/download/it/datasetv2.zip; then
        echo -e "${RED}❌ Download failed!${NC}"
        echo "Please check your internet connection and try again."
        exit 1
    fi
    
    echo "Extracting dataset..."
    if ! unzip -q datasetv2.zip; then
        echo -e "${RED}❌ Extraction failed!${NC}"
        exit 1
    fi
        
    # Elimina il train_raw.jsonl estratto dallo zip e ripristina quello corretto
    echo "📝 Restoring correct train_raw.jsonl..."
    if [ -f "train_raw.jsonl" ]; then
        rm train_raw.jsonl
        echo -e "${GREEN}✓${NC} Removed extracted train_raw.jsonl"
    fi
    
    if [ -f "train_raw_correct.jsonl" ]; then
        mv train_raw_correct.jsonl train_raw.jsonl
        echo -e "${GREEN}✓${NC} Restored train_raw_correct.jsonl as train_raw.jsonl"
    fi
    
    NUM_SAMPLES=$(wc -l < train_raw.jsonl)
    echo -e "${GREEN}✓${NC} Dataset ready: $NUM_SAMPLES samples"
else
    # Se abbiamo saltato il download, rinomina comunque train_raw_correct.jsonl in train_raw.jsonl
    if [ -f "train_raw_correct.jsonl" ]; then
        mv train_raw_correct.jsonl train_raw.jsonl
        echo -e "${GREEN}✓${NC} Using existing dataset"
        NUM_SAMPLES=$(wc -l < train_raw.jsonl)
    fi
fi

echo ""

# Make scripts executable
chmod +x *.sh *.py 2>/dev/null || true

# Calculate configuration
BATCH_SIZE=10
if [ "$NUM_GPUS" -ge 8 ]; then
    GRAD_ACCUM=1
    EPOCHS=3
    EST_TIME="~1.5 hours"
    EST_COST="~\$1.50"
elif [ "$NUM_GPUS" -ge 4 ]; then
    GRAD_ACCUM=2
    EPOCHS=5
    EST_TIME="~2.5 hours"
    EST_COST="~\$1.20"
else
    BATCH_SIZE=8
    GRAD_ACCUM=4
    EPOCHS=8
    EST_TIME="~6-8 hours"
    EST_COST="~\$2.00-3.00"
fi

EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

# Pre-download models (optional)
echo "=========================================="
echo "🤖 Pre-downloading models..."
echo "=========================================="
echo ""
echo "This will download ~4GB to cache. Skip if you want to save bandwidth."
echo "(Models will be downloaded automatically during training if skipped)"
echo ""

read -t 10 -p "Download now? (y/N, auto-skip in 10s): " -n 1 -r REPLY || REPLY='n'
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading models..."
    python3 << 'EOF'
from transformers import AutoConfig
from qwen_tts import Qwen3TTSTokenizer
import sys

try:
    print('  • Qwen3-TTS-12Hz-1.7B-Base...', end=' ')
    sys.stdout.flush()
    AutoConfig.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base')
    print('✓')
    
    print('  • Qwen3-TTS-Tokenizer-12Hz...', end=' ')
    sys.stdout.flush()
    Qwen3TTSTokenizer.from_pretrained('Qwen/Qwen3-TTS-Tokenizer-12Hz', device_map='cpu')
    print('✓')
    
    print('\n✓ Models cached successfully')
except Exception as e:
    print(f'\n⚠ Error caching models: {e}')
    print('Models will be downloaded during training')
EOF
else
    echo "Skipped - models will download on first use"
fi

echo ""

# Summary
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📊 Configuration Summary:"
echo "  • Working directory: $WORK_DIR"
echo "  • GPUs detected: $NUM_GPUS"
echo "  • Dataset samples: $NUM_SAMPLES"
echo "  • Batch size per GPU: $BATCH_SIZE"
echo "  • Gradient accumulation: $GRAD_ACCUM"
echo "  • Effective batch size: $EFFECTIVE_BATCH"
echo "  • Training epochs: $EPOCHS"
echo ""
echo "⏱️  Estimated Training:"
echo "  • Time: $EST_TIME"
echo "  • Cost: $EST_COST"
echo ""
echo "=========================================="
echo "🚀 Ready to Start Training!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. (Optional) Validate your dataset:"
echo "   python3 validate_dataset.py --jsonl train_raw.jsonl"
echo ""
echo "2. Start training:"
echo "   ./train_vast_optimized.sh"
echo ""
echo "   Or with tmux (recommended to survive disconnects):"
echo "   tmux new -s training"
echo "   ./train_vast_optimized.sh"
echo "   # Detach: Ctrl+B then D"
echo "   # Reattach: tmux attach -t training"
echo ""
echo "3. Monitor training (in another terminal):"
echo "   ./monitor_training.sh"
echo ""
echo "4. View Tensorboard (optional):"
echo "   tensorboard --logdir output_italian_tts --host 0.0.0.0 --port 6006"
echo ""
echo "5. After training completes, download your checkpoint:"
echo "   tar -czf checkpoint.tar.gz output_italian_tts/checkpoint-best"
echo "   scp -P <PORT> root@<IP>:$WORK_DIR/checkpoint.tar.gz ."
echo ""
echo "=========================================="
echo ""
echo "💡 Pro Tips:"
echo "  • GPU usage should be >90%: nvidia-smi -l 1"
echo "  • Watch logs live: tail -f output_italian_tts/training_*.log"
echo "  • Training continues if you disconnect (use tmux!)"
echo "  • Stop instance when done to avoid charges"
echo ""
echo "🆘 Having issues? Check:"
echo "  • https://github.com/OhMySimo/Qwen3-TTS-finetuning/issues"
echo "  • README: cat README.md"
echo ""
echo "Happy training! 🎉"
echo "=========================================="
