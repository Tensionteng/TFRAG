# 主要代码改动
## faiss构建索引
1. 目录: `models/Memory.py`
2. 单卡/多卡环境: 
   1. 路径: `exp/exp_short_term_forecasting.py`
   2. 具体代码:
        ```python
        class Exp_Long_Term_Forecast(Exp_Basic):
            def __init__(self, args):
                super(Exp_Long_Term_Forecast, self).__init__(args)
                if args.use_rag:
                    self.memory_bank = MemoryBankWithRetrieval(
                        seq_len=args.seq_len,
                        dim=args.enc_in,
                        pred_len=args.pred_len,
                        use_gpu=True,
                        gpu_index=1, # 单卡环境修改为0，多卡环境可以指定任意显卡索引
                    )
                    self.num_retrieve = args.num_retrieve
        ```

## 模型改动
1. 目录: `models/PatchTST.py` & `models/iTransformer.py`
2. 在预测头代码处加入两个预测头，分别用于预测均值和方差，具体代码:
   - 预测头
    ```python
    self.action_mean = nn.Sequential(
        nn.Linear(configs.pred_len, configs.pred_len, bias=True),
        # nn.Tanh(),
    )
    self.action_logstd = nn.Sequential(
        nn.Linear(configs.pred_len, configs.pred_len, bias=True),
        nn.Softplus(),
    )
    ```
    - 前向传播
    ```python
    def get_dist(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 有归一化和未归一化的版本，要做对比试验
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape

        x_enc = x_enc.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]
        action_mean = self.action_mean(x_enc).permute(0, 2, 1)[:, :, :N]
        action_logstd = self.action_logstd(x_enc).permute(0, 2, 1)[:, :, :N]
        action_logstd = torch.clamp(action_logstd, max=-1)
        action_std = torch.exp(action_logstd)  # [B, L, D]

        action_mean = action_mean * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        action_mean = action_mean + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        if torch.isnan(action_std).any() or torch.isinf(action_std).any():
            print("action_std 中有无效值")
        if (action_std <= 0).any():
            print("标准差非正")

        return torch.distributions.Normal(action_mean, action_std)
    ```
3. 工程上的trick: 
   - 在初始化预测头的时候，将`action_mean`初始化为单位阵，将`action_logstd`初始化为零矩阵，具体代码:
   ```python
    self.action_mean[0].weight = nn.Parameter(torch.eye(configs.pred_len))
    ```
    - 为了防止方差太大，需要对方差进行clip，具体代码:
    ```python
    action_logstd = torch.clamp(action_logstd, max=-1)
    action_std = torch.exp(action_logstd)
    ```
## 训练过程改动
1. 目录: `exp/exp_long_term_forecasting.py`
2. GRPO的训练方法，目前是将一个batch当作一个样本进行训练，例: 输入为[32, 96, 7]，rewards为[32, ]


## 需要做的消融实验
1. 所有模型和数据集有无rag的结果，目前只改了patchtst和itranformer，其他模型暂未更改源码
2. 是否需要两个trick，即初始化单位阵和clip方差
3. 如果使用了第一个trick，模型刚开始训练的时候会出现所有reward都是2的情况，GRPO之后advantage都是0，对模型的训练不起作用，可以考虑[DAPO](https://arxiv.org/abs/2503.14476)的dynamic sample，即过滤掉[32, ]里面的数字都是一样的样本，反复采样直到设定的数字
4. 由于存储了所有train dataset，检索到最相似的一定是自己，其对应的gt和batch_y一致，造成信息泄露，考虑去掉
5. 再想想怎么将训练时的能力转换到推理中