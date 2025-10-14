# Prison Escape - Quick Reference

## 🚀 Getting Started

1. **Install GAMA Platform** and launch server:
   ```bash
   gama-headless.bat -socket 1001  # Windows
   ./gama-headless.sh -socket 1001 # Linux/macOS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training**:
   ```bash
   # Quick test (recommended first)
   python run_training.py train --config quick
   
   # Full PPO training
   python run_training.py train
   
   # Advanced Q-Learning with anti-static mechanisms
   python improved_train_prison_escape.py
   ```

## 📁 Main Files

| File | Purpose | Algorithm |
|------|---------|-----------|
| `train_prison_escape.py` | Standard training | PPO |
| `improved_train_prison_escape.py` | Advanced training | Q-Learning + |
| `evaluate_agents.py` | Basic evaluation | - |
| `improved_evaluate_agents.py` | Advanced evaluation | - |
| `run_training.py` | Training utility | - |
| `image_viewer.py` | Visualization | - |

## 🔧 Configuration

- **Main config**: `config.py`
- **GAMA models**: `controler.gaml`, `PrisonEscape.gaml`
- **Dependencies**: `requirements.txt`

See **README.md** for complete documentation.