dlphys/utils/types.py 

types

Tensor: torch.Tensor 的类型别名（全局统一用法）。

Metrics: Dict[str, float]（训练/评估指标的标准返回格式）。

Batch: Any（允许 tuple/dict/nested 的通用 batch 类型）。

JSONDict: Dict[str, Any]（配置/记录用的字典类型）。

dlphys/utils/device.py 

device

get_device(device="cuda") -> torch.device：解析设备字符串并在不可用时安全回退到 CPU。

to_device(batch, device) -> Any：递归地把 batch 里的张量搬到指定 device（支持 tuple/list/dict 嵌套）。

dlphys/utils/io.py 

io

ensure_dir(path) -> Path：确保目录存在（mkdir -p 语义）并返回 Path。

save_json(path, obj, indent=2)：以 UTF-8 保存 JSON（自动创建父目录）。

load_json(path) -> Any：以 UTF-8 读取 JSON。

append_jsonl(path, record)：向 .jsonl 追加一条记录（每行一个 JSON）。

dlphys/utils/seed.py 

seed

set_seed(seed, deterministic=False)：同时设置 python/numpy/torch（含 CUDA）随机种子；可选开启尽量确定性。

seed_worker(worker_id)：配合 DataLoader 多进程 worker 的确定性初始化函数。

dlphys/utils/time.py 

time

now_str(fmt="%Y%m%d-%H%M%S", use_utc=False) -> str：生成时间戳字符串（本地或 UTC）。

human_time(seconds) -> str：把秒数格式化成人类可读的 Xs / Xm Ys / Xh Ym Zs。

Timer（dataclass）：

start()：开始计时并返回自身（可链式调用）。

elapsed() -> float：返回从 start 起的耗时秒数。

支持 with Timer() as t: 上下文用法。

dlphys/utils/loggers.py 

loggers

_resolve_path(p) -> Path：把路径展开并 resolve 成绝对路径（内部使用）。

clear_handlers(logger)：关闭并移除该 logger 的所有 handlers（Notebook/Windows 防文件锁）。

close_file_handlers(logger)：只关闭并移除 FileHandler，保留 StreamHandler。

get_logger(name="dlphys", level=INFO, log_file=None, fmt=..., datefmt=..., reset=False) -> Logger：Notebook 友好的 logger 工厂；避免重复 handler，并在 log_file 改变时替换文件输出；reset=True 强制重置配置。

set_level(logger, level)：同时设置 logger 和所有 handler 的 level。

dlphys/utils/hooks.py 

hooks

ActivationRecord（dataclass）：保存 hook 抓到的激活；activations[name] -> list[Tensor]。

clear()：清空缓存。

add(name, x)：追加一次激活。

last(name)：取最后一次激活。

keys()：返回已缓存的模块名列表。

HookHandleManager：统一管理 hook handle，remove_all() 一键移除。

ActivationCacher：注册 forward hook 抓层输出（支持按模块名或按类型筛选）。

register()：注册 hooks（只注册一次）。

remove()：移除 hooks。

clear()：清空已缓存激活。

支持 with ActivationCacher(...) as ac: 自动注册/清理。

dlphys/utils/stats.py 

stats

_to_numpy(x) -> np.ndarray：把 tensor/array/scalar 统一转成 numpy（内部使用）。

RunningMeanVar（dataclass）：Welford 在线均值/方差（数值稳定）。

update(x)：更新统计量。

var / std：返回样本方差/标准差（不足样本时为 None）。

reset()：重置状态。

RunningMoments（dataclass）：在线统计到四阶矩（mean/var/skew/kurtosis_excess）。

update(x)：更新（输入会 flatten 成 1D）。

skew / kurtosis_excess：偏度与超额峰度（不足样本时为 None）。

reset()：重置状态。