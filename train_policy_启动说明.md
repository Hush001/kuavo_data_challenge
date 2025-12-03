# train_policy.py 启动指南（基于多卡 A800 集群）

本文档说明如何基于提供的 `train.sh` 参数启动 `train_policy.py` 进行 ACT 策略的分布式训练。

## 1. 环境准备
1. 确保已在集群上加载需要的 GPU 资源（示例脚本申请了 4 张 A800）。
2. 激活训练所需的 Conda 环境：
   ```bash
   conda activate ooo
   ```
3. 确保 `PYTHONPATH` 包含项目根目录和 `third_party/lerobot/src`：
   ```bash
   export PYTHONPATH="/share/home/u21020/krm/leju/kuavo_data_challenge/third_party/lerobot/src:/share/home/u21020/krm/leju/kuavo_data_challenge:$PYTHONPATH"
   ```

## 2. 关键配置（已写入 `configs/policy/act_config.yaml`）
- `policy_name`: `act`
- `training.batch_size`: `64`（总 batch，DDP 会自动按 GPU 数划分）
- `training.max_epoch`: `3000`
- `training.num_workers`: `8`
- `training.output_directory`: `outputs`
- `training.resume`: `true`，`training.resume_timestamp`: `run_xxxx`
- `repoid`: `['lerobot1-200', 'lerobot201-400', ... , 'lerobot1801-2000']`
  - 支持在命令行里传入字符串形式（例：`repoid="['lerobot1-200','lerobot201-400']"`），脚本会自动解析为列表。
- `root`: `/ssdfs/datahome/u21020/1`
- `hydra.run.dir`: `.`（日志和配置保存在当前运行目录）

### 如何设置本地数据目录
`root` 需要指向**包含所有分片目录的上级路径**，每个分片形如：

```
/ssdfs/datahome/u21020/1/
  ├─ lerobot1-200/
  │   └─ lerobot/
  │       ├─ data/
  │       ├─ images/
  │       └─ meta/
  ├─ lerobot201-400/
  │   └─ lerobot/...
  └─ ...
```

只要把 `root` 设置成 `lerobot*` 分片的共同父目录即可，例如你的数据结构可以直接使用 `root: /ssdfs/datahome/u21020/1`。如有自定义路径，保持同样的层级（`<root>/<分片名>/lerobot/{data,images,meta}`）即可被正确读取。

如需调整数据路径或重训策略，可直接编辑 `configs/policy/act_config.yaml`。

## 3. 启动命令示例
以下命令与 `train.sh` 保持一致，可在项目根目录执行（脚本路径位于 `kuavo_train/train_policy.py`，默认加载 `configs/policy/act_config.yaml`）：
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
MASTER_PORT=29500

# 启动分布式训练
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  train_policy.py \
  policy_name=act \
  training.batch_size=64 \
  training.max_epoch=3000 \
  training.num_workers=8 \
  training.output_directory=$(pwd)/outputs \
  hydra.run.dir=. \
  training.resume=true \
  training.resume_timestamp="run_xxxx"
```
> 提示：如果需要自定义数据集分片或其他 Hydra 参数，可在命令末尾继续追加 `key=value` 形式的覆盖。

## 4. 日志与模型输出
- 模型权重与日志默认保存在 `./outputs` 下。
- 断点续训会从 `outputs/<task>/<method>/run_xxxx` 中加载最近一次保存的权重。

祝训练顺利！
