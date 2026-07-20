# Goal 1B：B300 MoE Decode Fast Path 实现报告

审计日期：2026-07-20（Asia/Shanghai）

## 1. 结论

本仓库现已具备 Goal 1B 的公开 Python 接口、显式后端选择、纯 PyTorch
正确性实现、稳定路由元数据、可复用 workspace、BF16 CUDA 分阶段源码、
严格的 `sm_103a` 构建预检、目标机测试矩阵和八阶段 benchmark 工具。

但当前开发机只有 PyTorch `2.12.0+cpu`，没有 NVIDIA GPU、CUDA、`nvcc`
或 CUTLASS。因此当前实际执行后端只有 `torch`；`cutlass_bf16` 源码未在
B300 上编译或运行，`cutlass_nvfp4` 明确不可用。Goal 1B 的 B300 正确性和
性能验收尚未完成，本文不把源码存在、编译参数或跳过的测试当作硬件证据。

| 项目 | 当前状态 | 证据边界 |
|---|---|---|
| Python API 与输入合同 | 已实现 | CPU 测试已执行 |
| `torch` 后端 | 已实现 | 当前主机实际执行，`used_fallback=false` |
| 稳定路由 | Python 已验证；CUDA 有源码 | CUDA 版本尚未在 GPU 执行 |
| `cutlass_bf16` | 有完整分阶段调用源码 | 未编译、未运行、未做 B300 数值/性能验证 |
| `cutlass_nvfp4` | 未实现 | 明确报错，不回退到 BF16 或 torch |
| B300 benchmark | 工具已实现 | 当前没有 B300 数据，不能给出瓶颈排名 |

更完整的主机审计见 `docs/goal_1b_environment.md`。

## 2. 实现组成

- `fastpath/b300_moe.py`：公开 API、后端能力检查、workspace 和 BF16 分阶段编排。
- `fastpath/reference.py`：消费显式路由的 PyTorch 正确性实现与输入验证。
- `csrc/routing_kernels.cu`：计数、offset 和稳定双向映射。
- `csrc/permutation_kernels.cu`：按 expert-major 顺序 gather hidden states。
- `csrc/grouped_gemm_sm103.cu`：BF16 grouped GEMM 的 ATen 内部接口适配层。
- `csrc/moe_fastpath.cu`：独立 SwiGLU CUDA kernel。
- `csrc/combine_kernels.cu`：反向映射、路由权重和 FP32 累加 combine。
- `csrc/quantization_kernels.cu`：真实报告 NVFP4 当前不可用。
- `csrc/bindings.cpp`：分阶段 pybind API 与构建元数据。
- `fastpath/build.py`、`setup.py`、`scripts/build_goal_1b.sh`：严格目标构建入口。
- `benchmark/benchmark_goal_1b.py`：八阶段 cold/steady-state 计时与 CSV 输出。

## 3. 数据流

`b300_moe_forward` 直接消费上游给出的 expert indices 和 weights；top-k logits
选择不属于 forward 本身。benchmark 为测量完整请求，另行把 top-k routing
计入 `routing_us` 和 `total_us`。

```mermaid
flowchart LR
    H[hidden_states<br/>T x H]
    R[expert_indices / expert_weights<br/>T x K]
    M[计数 + offsets<br/>稳定双向映射]
    P[expert-major permutation<br/>A x H, A=T*K]
    G1[BF16 grouped GEMM<br/>A x H · E x H x 2I]
    S[SwiGLU<br/>silu(gate) * up<br/>A x I]
    G2[BF16 grouped GEMM<br/>A x I · E x I x H]
    C[reverse mapping + weighted combine<br/>T x H]
    O[output<br/>T x H]

    R --> M --> P
    H --> P --> G1 --> S --> G2 --> C --> O
    R --> C
```

## 4. API 与张量布局

公开调用形式为：

```python
b300_moe_forward(
    hidden_states,
    expert_indices,
    expert_weights,
    gate_up_weight,
    down_weight,
    *,
    num_experts,
    top_k,
    quant_mode=None,
    backend="torch",
    workspace=None,
    return_metadata=False,
)
```

张量合同：

| 张量 | 布局 | 含义 |
|---|---|---|
| `hidden_states` | `[T, H]` | 输入 token states |
| `expert_indices` | `[T, K]`, `int64` | token-major、rank-minor 的显式路由 |
| `expert_weights` | `[T, K]` | 原样用于 combine，不重新归一化 |
| `gate_up_weight` | `[E, 2I, H]` | 前 `I` 行为 gate，后 `I` 行为 up |
| `down_weight` | `[E, H, I]` | down projection |
| 返回值 | `[T, H]` | 加权聚合结果 |

`K` 支持 `1/2/4/8`；公共合同要求所有输入同设备、浮点输入同 dtype 且
contiguous。公共 Python 验证接受 `float32/float16/bfloat16`，但已编译的
`cutlass_bf16` 路径只接受 BF16，且当前 CUDA 路由实现限制 `E <= 256`。
编译后端不支持 autograd；`torch` 后端在不传 workspace 时保留 autograd。

可选 metadata 至少真实返回：`backend`、`architecture`、`quant_mode` 和
`used_fallback`，并附带 counts、offsets、双向映射和 workspace 统计。
请求编译后端是一项精确请求：硬件、dtype、扩展或入口点不匹配时直接失败。
编译路径还附带 `build` 字典，分别报告 wrapper 编译架构、外部 CUTLASS header
版本、PyTorch C++ 版本和实际调用的 ATen provider，避免把这些来源混为一谈。

## 5. 稳定路由与空 expert

令扁平 assignment 编号为 `a = token * K + rank`。按 expert id 分组后：

- `permutation[p]`（即 `permuted_to_assignment`）给出分组行 `p` 对应的 `a`；
- `reverse_mapping[a]`（即 `assignment_to_permuted`）给出 `a` 对应的分组行；
- 两者互为逆映射，同一 expert 内保持原 token/rank 顺序；
- `expert_offsets[e:e+2]` 描述 expert `e` 的行区间；空 expert 的区间长度为零。

CUDA 源码先用 atomic add 统计每个 expert 的数量，再生成 exclusive offsets；
随后每个 expert 由一个线程按 `a=0..A-1` 扫描，因此映射是确定且稳定的。
该实现支持动态计数、重复 expert 路由和空 expert，但映射阶段复杂度为
`O(E*A)`，是为 decode 小批量优先保证正确性的初始实现，并非已调优结论。

## 6. 两段 grouped GEMM、SwiGLU 与 combine

BF16 编排使用 `A=T*K` 个分组后的 assignment：

1. `permute_out` 生成 `[A,H]`。
2. 第一段 grouped GEMM 生成 `[A,2I]`。
3. SwiGLU 把前半 gate 与后半 up 计算为 `silu(gate) * up`，得到 `[A,I]`。
4. 第二段 grouped GEMM 生成 `[A,H]`。
5. `combine_out` 按反向映射读取各 rank 结果，乘调用方原始 routing weight，
   在每个输出元素内以 FP32 累加，再转换回输出 dtype。

公共权重无需持久化 repack。第一段把 `[E,2I,H]` 零拷贝 transpose 为
`[E,H,2I]`；第二段把 `[E,H,I]` transpose 为 `[E,I,H]`。适配器要求该
`[E,K,N]` view 的 `stride(1)==1`。每个 expert 的累计结束 offset 会从
`int64` 转换并缓存到 grouped GEMM 所需的 `int32[E]`。

## 7. CUTLASS/SM103 BF16 实现的准确描述

`csrc/grouped_gemm_sm103.cu` 调用 PyTorch 内部 CUDA 接口
`at::cuda::detail::bf16bf16_grouped_mm(..., out)`。CUDA-enabled PyTorch 中该
实现是 CUTLASS-backed grouped GEMM，并允许写入调用方预分配输出。

外部 CUTLASS checkout 在本仓库中的作用是编译期门槛：源码包含
`cutlass/arch/arch.h` 与 `cutlass/version.h`，静态检查 CUTLASS `>=4.3.1`
以及 `cutlass::arch::Sm103`。构建只发出
`-gencode=arch=compute_103a,code=sm_103a`，不产生其他架构 fallback。

这不是仓库自有的、直接实例化并可独立调参的 CUTLASS kernel。它依赖 ATen
内部 ABI，可能随 PyTorch 版本变化；外部 CUTLASS 静态检查也不能证明
ATen 最终选中的具体 tile/schedule。必须在目标 PyTorch/CUDA 组合上完成编译、
运行、profiler 或二进制检查后，才能把它认定为已验证的 B300 kernel。
`build_info()` 因此把 `wrapper_compiled_architecture=sm_103a`、
`external_cutlass_headers_version` 与 `grouped_gemm_provider_architecture`
分开；后者属于预编译 PyTorch binary。预检还要求该 binary 的
`torch.cuda.get_arch_list()` 明确包含 SM103。

## 8. 数据类型、量化与 scale

| 路径 | 输入/权重 | 中间与累加 | 量化状态 |
|---|---|---|---|
| `torch` | FP32/FP16/BF16 同 dtype | 由 PyTorch reference 决定 | 不量化 |
| `cutlass_bf16` | BF16 | GEMM 由 ATen/CUTLASS 决定；SwiGLU 与 combine 显式转 FP32 运算 | 不量化 |
| `cutlass_nvfp4` | 未定义真实运行合同 | 未实现 | 不可用 |

当前没有 E2M1/NVFP4 payload 打包，没有 activation/weight scale 张量，没有
block-scale 布局，也没有 scale 生命周期或校准策略。扩展能力函数固定返回
`false`，并说明 “SM103 NVFP4 grouped GEMM adapter is not compiled”。Python
还要求未来实现同时提供 `quantize_nvfp4_out` 和 `grouped_gemm_nvfp4_out`；
即使能力标志误报为 true，也不会偷偷执行 BF16。故当前 `quant_mode=nvfp4`
只能明确失败，`used_fallback` 不会被伪造为成功。

BF16/torch benchmark 的 `quantization_us` 标记为 `not_applicable`，数值为零；
NVFP4 根本无法进入计时，工具不会把其缺失错误记录成零耗时。

## 9. Workspace 与生命周期

`B300MoEWorkspace` 缓存 counts、offsets、两种映射、permuted token/expert/weight、
`int32` GEMM offsets、permuted hidden、gate-up 输出、SwiGLU 输出和 expert 输出。
layout signature 包含设备、dtype、E、K、H、I、backend 与 quant mode。

- 容量足够且 layout 不变时复用原指针；扩容采用至少翻倍策略。
- workspace 绑定首次使用设备，不允许静默跨设备重绑。
- CUDA stream 间通过 completion event 排序，并对缓冲区调用 `record_stream`。
- CUDA graph capture 期间禁止扩容/换 layout；调用方应在 capture 前 reserve。
- 返回 output 不别名 scratch；metadata 中路由张量会 clone，后续复用不改旧结果。
- 显式 workspace 为 inference-only；并发访问由 workspace lock 串行化。

正常编译路径通过 `routing_metadata_out` 直接写入 workspace；保留的 allocating
`routing_metadata` 只用于兼容接口。当前仍非完全 allocation-free：最终 output
每次新建；无显式 workspace 的调用会创建临时 workspace；更重要的是 ATen
内部 grouped-MM 会为 argument buffer 与 CUTLASS workspace 使用其自身 allocator，
两段 GEMM 均不受 `B300MoEWorkspace` 管理。因此当前不能声称 descriptor 已缓存或
CUDA graph capture-ready；预 reserve 只保证本仓库管理的缓冲区和 completion event
提前创建，目标机仍需验证 ATen 内部分配的 capture 行为。

## 10. 正确性验证

CPU 测试覆盖独立逐 token/逐 expert oracle、`K=1/2/4/8`、不归一化及负 routing
weights、重复路由、空 expert、映射稳定性/互逆性、FP16/BF16 reference、严格
输入校验、metadata、workspace 复用/扩容/设备绑定，以及无 workspace autograd。

本次执行：

```text
python -m pytest -q tests -k goal_1b
50 passed, 94 skipped, 51 deselected in 2.84s

python -m pytest -q
86 passed, 109 skipped in 2.78s
```

跳过项包含要求 CUDA 扩展与 CC `(10,3)` 的 B300 测试。目标测试矩阵已定义
`T=1/2/4/8/16/32/64`、`K=1/2/4/8` 和 uniform/hotspot/zipf，并使用
`atol=0.12, rtol=0.06` 对照 torch；在本机被 skip 不等于通过。

## 11. Benchmark 口径

CSV 对每个 case 输出以下八个精确 stage 的 P50/P90/P99（微秒）：

1. `routing_us`
2. `permutation_us`
3. `quantization_us`
4. `gate_up_gemm_us`
5. `swiglu_us`
6. `down_gemm_us`
7. `combine_us`
8. `total_us`

另记录单次 `cold_start_us`；steady-state 先 warmup，再重复采样。CUDA 使用同步
CUDA events，CPU 使用高分辨率 wall clock。随机输入、权重构造和设备搬运位于
计时外。CSV 记录 requested/actual backend、fallback、GPU/CC、PyTorch/CUDA/
CUTLASS 版本、shape、workload、dtype、quant mode 与计时方法。

口径细节：`routing_us` 是从 router logits 执行 top-k；`permutation_us` 同时包含
从显式 indices 建立 counts/offsets/mappings 以及 gather。`total_us` 包含 top-k
routing 和完整 forward。各 stage 独立计时，不能简单相加后与 total 做严格等式。

当前只验证了 CPU `torch` benchmark/CSV 合同；没有任何 B300 延迟数据。
请求不可用的编译后端默认报错，只有显式 `--skip-unavailable` 才会打印 skip；
不会把 torch 行冒充为 CUTLASS 行。

本机真实 CPU smoke 使用 `T=1,E=8,K=1,H=8,I=12`、uniform、3 次 warmup、
20 次 steady-state 重复。实际 metadata 为 `backend=torch`、`quant_mode=none`、
`used_fallback=false`；首次 forward（输入/权重已构造后）为 `1228.700 us`：

| stage | P50 (us) | P90 (us) | P99 (us) |
|---|---:|---:|---:|
| routing | 4.600 | 5.210 | 5.624 |
| permutation | 56.050 | 88.950 | 183.990 |
| quantization (`not_applicable`) | 0.000 | 0.000 | 0.000 |
| gate/up GEMM | 8.750 | 9.220 | 9.724 |
| SwiGLU | 5.650 | 5.810 | 5.900 |
| down GEMM | 8.500 | 8.810 | 9.062 |
| combine | 6.750 | 6.900 | 6.981 |
| total | 213.650 | 392.740 | 577.429 |

这些数值只是当前 Windows CPU reference 的一次小样本，不代表 B300 性能，也不
用于推断瓶颈。`configs/b300.yaml` 带有 `requires_cuda: true`，默认 CPU 调用会在
分配约 42 GiB BF16 权重前明确拒绝。

## 12. 性能瓶颈：证据与限制

当前没有 B300、Nsight Systems/Compute trace、kernel duration、occupancy、带宽或
Tensor Core 指标，因此不能声称某阶段是实测瓶颈，也不能报告相对 Goal 1A 的
加速比。CPU reference 计时不代表 GPU decode 行为。

仅从源码可列出待验证假设：稳定 mapping 的 `O(E*A)` 扫描、单线程 offsets、
每次 routing 临时分配与 workspace copy、多个独立 elementwise/memory kernels、
以及 tiny-M grouped GEMM 的 launch/调度开销都可能影响 decode 延迟。这些应由
目标机八阶段数据和 profiler 证实，而非在本报告中排序。

## 13. 已知局限与未完成项

- 当前主机：Windows 11、Python 3.12.10、PyTorch `2.12.0+cpu`；无 CUDA GPU、
  CUDA toolkit、`nvcc`、CUTLASS 或可用 CUDA host toolchain。
- `python -m fastpath.build --check-only` 在本机按预期返回 `FAILED`；实际编译
  architecture 为 none，不能因源码写有 `sm_103a` 而报告已构建。
- BF16 适配层依赖 ATen 内部 API，尚未证明能与目标机 PyTorch ABI 编译链接。
- 未验证内部 grouped GEMM 对空组、极小 M、非规则 H/I 和对齐约束的行为；当前
  没有 padding/对齐补偿路径。
- CUDA 路由、SwiGLU、combine、stream ordering 和 graph capture 均只有源码或
  目标测试，尚未在 GPU 执行。
- 预分配 workspace 尚不等于 CUDA graph capture-ready；ATen grouped-MM 的内部
  argument/workspace allocation 必须在目标机单独验证。
- 编译后端无 backward；expert 数限制 256；T 必须大于零。
- 尚未实现真正 NVFP4 datatype、双层/块 scale、量化 kernel 或 NVFP4 GEMM。
- 尚未取得 B300 cold/steady-state 数据、profiler 证据或验收阈值结论。

目标机复现入口：

```bash
export CUDA_HOME=/path/to/cuda-13.x
export CUTLASS_PATH=/path/to/cutlass
python -m fastpath.build --check-only
bash scripts/build_goal_1b.sh
python -m pytest -q tests/test_goal_1b_cuda.py
MOE_RUN_B300_LARGE=1 python -m pytest \
  tests/test_goal_1b_cuda.py -k b300_config_target_shape -v
python benchmark/benchmark_goal_1b.py \
  --config configs/b300.yaml --device cuda --backend cutlass_bf16
```

只有预检显示真实 CC 10.3 GPU、编译成功、`build_info()` 报告 Goal 1B/
`sm_103a`/无 fallback、CUDA 测试通过且 benchmark 产生真实 CUTLASS 行后，
才能关闭 BF16 的 B300 验收缺口。

## 14. Goal 1C 建议

1. 先在固定 B300 + CUDA 13.x + CUDA-enabled PyTorch + CUTLASS 版本组合上完成
   BF16 bring-up，保存完整编译日志、`build_info()`、correctness 和 profiler 证据。
2. 用仓库自有、直接实例化的 SM103 CUTLASS MoE/grouped GEMM adapter 替换 ATen
   内部接口，显式记录 tile、cluster、schedule、alignment 和累加类型。
3. 实现真实 NVFP4 E2M1 activation/weight packing 与 CUTLASS 所需 scale 层级，
   把 scale shape、dtype、布局、生成时机和误差容限纳入公共合同与 metadata。
4. 将稳定路由改为并行 histogram/scan/stable scatter，复用路由输出并消除
   `routing_metadata` 临时分配及 int64-to-int32 copy。
5. 评估 permutation+quantization、SwiGLU epilogue、reverse+weighted combine 的
   融合，优先以 `T<=64` 的真实 stage P99 决定工程顺序。
6. 为常见 H/I/E/K 建立对齐与 padding 策略，预热 kernel selection，并完成
   CUDA graph capture、跨 stream、空 expert 和极端热点分布压力测试。
7. 对 uniform/hotspot/zipf 分别采集 Nsight 指标，报告 cold 与 steady-state、
   P50/P90/P99、有效带宽、Tensor Core 利用率和每阶段 launch 数。
8. 最后再加入 NVFP4 与 BF16 的同 shape 精度/延迟/显存对照；未通过真实
   NVFP4 capability 检查的结果不得命名为 NVFP4。
