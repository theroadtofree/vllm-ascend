# PDmix分布式边云协同推理
边云协同将大预言模型（LLM）的不同层分散部署在边侧（Edge/Master）和云测（Cloud/Slave）,通过边云协同完成推理
## 1 角色划分
边侧（Edge）执行模型的首层（Head）和尾层（Tail）,包括Embedding、首若干Transformer层、尾若干Transformer层、LM Head 等。默认情况为首1层尾1层
云测（Cloud）执行模型的中间层（Layers），即边侧首层之后，尾层之前的全部Transformer层。
## 2 执行过程
### 2.1 Prefill 阶段
1. Edge 首层执行：边侧执行head部分（embedding + 首层），生成中间hidden state
2. 边云数据传输：边侧通过HCCL isend 将 hidden state 异步发送给云测；云测通过irecv异步接收
3. Cloud 中间层执行：云测执行中间层，生成输出hidden state
4. 云边数据回传: 云测将结果 hidden state 发送回边侧。
5. Edge 尾层执行：边侧执行 tail 部分（尾层 + lm_head）,生成logits
### 2.2 Decode 阶段
流程与Prefill 类似，但处理的是单个token的迭代生成：
- Edge执行head -> 发送hidden -> CLoud 执行layers -> 返回 hidden -> Edge执行tail -> 采样

## 3 基于PDmix的边云协同推理版本
vllm仓和vllm-ascend仓已基于PDmix实现了边云协同推理，vllm服务启动命令参考如下：   

**边侧(rank0)：**  
vllm server Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 8060 \
    --master-addr 76.76.26.194 \
    --master-port 29501 \
    --served-model-name qwen3.6 \
    --trust-remote-code \
    --nnodes 2 \
    --node-rank 0 \
    --enable-edge-cloud \
    --edge-npu-count 2 \
    --cloud-npu-count 4 \
    --additional-config '{"edge_cloud_config":{"enble":true, "role":"edge", "edge_head_tail_layers":1}}' \
    --compilation-config '{"cudagraph_mode":"NONE", "cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
    --enable-edge-cloud-async-sched \
    --max-batch-last-depth 1 

**云侧(rank0)：**  
vllm server Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 8060 \
    --master-addr 76.76.26.194 \
    --master-port 29501 \
    --served-model-name qwen3.6 \
    --trust-remote-code \
    --nnodes 2 \
    --node-rank 1 \
    --enable-edge-cloud \
    --edge-npu-count 2 \
    --cloud-npu-count 4 \
    --additional-config '{"edge_cloud_config":{"enble":true, "role":"cloud", "edge_head_tail_layers":1}}' \
    --compilation-config '{"cudagraph_mode":"NONE", "cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
    --enable-edge-cloud-async-sched \
    --max-batch-last-depth 1 
