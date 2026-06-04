# PDmix分布式边云推理支持异步调度简要设计文档
# 1 现状分析
当前 PDmix 分布式边云推理中，一个 batch 的首层（Head）与尾层（Tail）执行在**同一个 step 内强耦合**：
1. **边侧 NPU 空转**：边侧 Worker 执行完 batch1 首层后，必须阻塞等待云侧返回 hidden state，再推进尾层；期间即使 Scheduler 已准备好 batch2 的 SchedulerOutput，也无法下发。
2. **云侧气泡大**：云侧执行两个 batch 之间，至少要等待 batch1 尾层执行时间 + batch2 首层执行时间 + 双向 hidden state 传输时间。

# 2 边侧异步调度整体设计
1. SchedulerOutput的首尾层执行解耦：SchedulerOutput执行完首层后要能返回EngineCore层，并将SchedulerOutput保存在队列batch_last[]中（scheduler新增）
2. 支持两个batch的首层的连续调度，即batch1的首执行后返回能马上执行batch2的首，减少边侧NPU空转时间；batch2首执行提前的同时云测batch1与batch2之间的执行间隔也就减少了，同步减少了云测NPU的空转时间
3. 两个batch的首层支持连续调度后，需要拓展hitten state的传输通道，从单通道变为双通道
# 3 边侧首尾调度算法设计

|batch类型|请求来源|
|:---:|:---:|
|batch_first|`waiting[]`、`running`|
|batch_last|`batch_last[]`|

|调度类型|优先级(数字小优先级高)|调度条件|
|:---:|:---:|:---:|
|batch_first|1|1、`batch_last[]`的长度小于2;<br>2、`waiting[]`或`running`不为空|
|batch_last|2|`batch_last[]`不为空|
|空batch|3|batch_first和batch_last的下发条件均不满足|

# 4 开发计划
|开发阶段|功能点|已开发|
|:---:|:---:|:---:|
|phase1|边侧首尾调度算法设计|❌|
|phase2|SchedulerOutput下发执行首层后返回EngineCore层|❌|
|phase3|拓展hitten state传输通道为双通道|❌|