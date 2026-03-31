#!/bin/bash
# ============================================================
# setup_openpi.sh
# 将草莓采摘项目的训练文件注入到 openpi 框架中
# 运行一次即可，之后就可以直接用 openpi 训练和推理
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENPI_DIR="$SCRIPT_DIR/openpi"

# ── 检查 openpi 是否存在 ──
if [ ! -d "$OPENPI_DIR" ]; then
    echo "错误: 没有找到 openpi 目录"
    echo "请先运行:"
    echo "  cd $SCRIPT_DIR"
    echo "  git clone https://github.com/Physical-Intelligence/openpi.git"
    echo "  cd openpi && uv sync"
    exit 1
fi

echo "项目目录: $SCRIPT_DIR"
echo "openpi 目录: $OPENPI_DIR"
echo ""

CONFIG_FILE="$OPENPI_DIR/src/openpi/training/config.py"

# ============================================================
# Step 1: 复制 strawberry_policy.py
# ============================================================
echo "[1/4] 复制 strawberry_policy.py ..."
cp "$SCRIPT_DIR/training/strawberry_policy.py" \
   "$OPENPI_DIR/src/openpi/policies/strawberry_policy.py"
echo "  ✓ → openpi/src/openpi/policies/"

# ============================================================
# Step 2: 复制 convert_to_lerobot.py
# ============================================================
echo "[2/4] 复制 convert_to_lerobot.py ..."
cp "$SCRIPT_DIR/data_conversion/convert_to_lerobot.py" \
   "$OPENPI_DIR/convert_to_lerobot.py"
echo "  ✓ → openpi/"

# ============================================================
# Step 3: 在 config.py 顶部添加 import
# ============================================================
echo "[3/4] 修改 config.py (import + DataConfig class + TrainConfig) ..."

if grep -q "import openpi.policies.strawberry_policy" "$CONFIG_FILE"; then
    echo "  ✓ import 已存在，跳过"
else
    sed -i '/import openpi.policies.libero_policy as libero_policy/a import openpi.policies.strawberry_policy as strawberry_policy' "$CONFIG_FILE"
    echo "  ✓ 添加了 import strawberry_policy"
fi

# ============================================================
# Step 4: 用 Python 注入 DataConfig class + TrainConfig
#         (sed 处理多行太脆弱，用 Python 更可靠)
# ============================================================

python3 << 'PYEOF'
import re, sys

config_path = sys.argv[1] if len(sys.argv) > 1 else ""
if not config_path:
    # 从环境读
    import os
    config_path = os.environ.get("CONFIG_FILE", "")

with open("OPENPI_CONFIG_PATH", "w") as f:
    pass  # placeholder

PYEOF

# 用 Python 脚本注入（更可靠）
python3 - "$CONFIG_FILE" << 'PYEOF'
import sys

config_path = sys.argv[1]

with open(config_path, "r") as f:
    content = f.read()

modified = False

# ── 注入 DataConfig class ──
if "LeRobotStrawberryDataConfig" not in content:
    dataconfig_code = '''

@dataclasses.dataclass(frozen=True)
class LeRobotStrawberryDataConfig(DataConfigFactory):
    """Custom data config for UR5e strawberry picking in Isaac Sim (3 cameras)."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/cam1": "observation.images.cam1",
                        "observation/cam2": "observation.images.cam2",
                        "observation/cam3": "observation.images.cam3",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[strawberry_policy.StrawberryInputs(model_type=model_config.model_type)],
            outputs=[strawberry_policy.StrawberryOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(6, -2)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            action_sequence_keys=("action",),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

'''
    # 在 RLDSDroidDataConfig 之前插入
    anchor = "class RLDSDroidDataConfig"
    if anchor in content:
        content = content.replace(anchor, dataconfig_code + anchor)
        modified = True
        print("  ✓ 添加了 LeRobotStrawberryDataConfig class")
    else:
        print("  ⚠ 找不到 RLDSDroidDataConfig 锚点，请手动添加 DataConfig")
else:
    print("  ✓ LeRobotStrawberryDataConfig 已存在，跳过")

# ── 注入 TrainConfig ──
if 'name="pi05_strawberry_3c"' not in content:
    trainconfig_code = '''    # Strawberry picking with UR5e in Isaac Sim (3 cameras)
    TrainConfig(
        name="pi05_strawberry_3c",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=10),
        data=LeRobotStrawberryDataConfig(
            repo_id="local/strawberry_picking_3c",
        ),
        batch_size=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        wandb_enabled=False,
    ),
'''
    # 在 _CONFIGS 列表的最后一个 ] 之前插入
    # 找最后一个独立的 ]
    lines = content.split('\n')
    insert_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == ']':
            insert_idx = i
            break

    if insert_idx is not None:
        lines.insert(insert_idx, trainconfig_code)
        content = '\n'.join(lines)
        modified = True
        print("  ✓ 添加了 pi05_strawberry_3c TrainConfig")
    else:
        print("  ⚠ 找不到 _CONFIGS 列表末尾，请手动添加 TrainConfig")
else:
    print("  ✓ pi05_strawberry_3c TrainConfig 已存在，跳过")

if modified:
    with open(config_path, "w") as f:
        f.write(content)

PYEOF

# ============================================================
# Step 5: 复制 norm stats
# ============================================================
echo "[4/4] 复制 norm stats ..."
if [ -d "$SCRIPT_DIR/assets/pi05_strawberry" ]; then
    mkdir -p "$OPENPI_DIR/assets"
    cp -r "$SCRIPT_DIR/assets/pi05_strawberry" "$OPENPI_DIR/assets/pi05_strawberry"
    echo "  ✓ → openpi/assets/pi05_strawberry/"
else
    echo "  ⚠ norm stats 不存在，后续需要运行 compute_norm_stats.py"
fi

# ============================================================
# 验证
# ============================================================
echo ""
echo "════════════════════════════════════════════════════"
echo " 验证"
echo "════════════════════════════════════════════════════"

OK=true
check() { if [ "$1" = "true" ]; then echo "  ✓ $2"; else echo "  ✗ $2"; OK=false; fi }

[ -f "$OPENPI_DIR/src/openpi/policies/strawberry_policy.py" ] && R=true || R=false; check $R "strawberry_policy.py"
[ -f "$OPENPI_DIR/convert_to_lerobot.py" ] && R=true || R=false; check $R "convert_to_lerobot.py"
grep -q "import openpi.policies.strawberry_policy" "$CONFIG_FILE" && R=true || R=false; check $R "config.py import"
grep -q "LeRobotStrawberryDataConfig" "$CONFIG_FILE" && R=true || R=false; check $R "config.py DataConfig"
grep -q 'name="pi05_strawberry_3c"' "$CONFIG_FILE" && R=true || R=false; check $R "config.py TrainConfig"

echo ""
if $OK; then
    echo "全部通过！"
    echo ""
    echo "验证 openpi 能加载:"
    echo "  cd $OPENPI_DIR"
    echo '  uv run python -c "import openpi.training.config; print(\"OK\")"'
else
    echo "有错误，请检查上方输出。"
    exit 1
fi
