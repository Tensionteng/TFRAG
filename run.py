import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np


def build_setting(args, ii):
    """Experiment id. Every field that changes results must appear here, or two
    different conditions will overwrite each other's results/ directory."""
    parts = [
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        f"ft{args.features}",
        f"sl{args.seq_len}",
        f"ll{args.label_len}",
        f"pl{args.pred_len}",
        f"dm{args.d_model}",
        f"nh{args.n_heads}",
        f"el{args.e_layers}",
        f"dl{args.d_layers}",
        f"df{args.d_ff}",
        f"fc{args.factor}",
        f"eb{args.embed}",
        f"lr{args.learning_rate}",
        f"bs{args.batch_size}",
        f"ep{args.train_epochs}",
        f"loss{args.loss}",
        f"seed{args.seed}",
        args.des,
        str(ii),
    ]
    if args.use_rag:
        parts += [
            "rag",
            f"g1{args.gamma_1}",
            f"g2{args.gamma_2}",
            f"k{args.num_retrieve}",
            f"ns{args.num_rl_samples}",
            f"ret{args.retrieval_mode}",
            f"excl{args.exclusion_radius}",
        ]
        if args.gamma_3:
            parts.append(f"g3{args.gamma_3}{args.distill_target}")
        if args.lambda_reg:
            parts.append(f"lam{args.lambda_reg}")
        if args.freeze_policy:
            parts.append("frozen")
        if args.detach_yhat:
            parts.append("detach")
        if args.reward_type != "discrete":
            parts.append(args.reward_type)
    if args.tag:
        parts.append(args.tag)
    return "_".join(str(p) for p in parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTh1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%%)"
    )

    # model define
    parser.add_argument(
        "--expand", type=int, default=2, help="expansion factor for Mamba"
    )
    parser.add_argument(
        "--d_conv", type=int, default=4, help="conv kernel size for Mamba"
    )
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        help="0: channel dependence 1: channel independence for FreTS model",
    )
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        help="method of series decompsition, only support moving_avg or dft_decomp",
    )
    parser.add_argument(
        "--use_norm",
        type=int,
        default=1,
        help="whether to use normalize; True 1 False 0",
    )
    parser.add_argument(
        "--down_sampling_layers",
        type=int,
        default=0,
        help="num of down sampling layers",
    )
    parser.add_argument(
        "--down_sampling_window", type=int, default=1, help="down sampling window size"
    )
    parser.add_argument(
        "--down_sampling_method",
        type=str,
        default=None,
        help="down sampling method, only support avg, max, conv",
    )
    parser.add_argument(
        "--seg_len",
        type=int,
        default=96,
        help="the length of segmen-wise iteration of SegRNN",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--gpu_type", type=str, default="cuda", help="gpu type"
    )  # cuda or mps
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )

    # metrics (dtw)
    parser.add_argument(
        "--use_dtw",
        type=bool,
        default=False,
        help="the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)",
    )

    # Augmentation
    parser.add_argument(
        "--augmentation_ratio", type=int, default=0, help="How many times to augment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
        help="global seed: seeds python/numpy/torch training AND augmentation. "
        "Vary this to produce multi-seed results (was hardcoded to 2021 before)",
    )
    parser.add_argument(
        "--jitter",
        default=False,
        action="store_true",
        help="Jitter preset augmentation",
    )
    parser.add_argument(
        "--scaling",
        default=False,
        action="store_true",
        help="Scaling preset augmentation",
    )
    parser.add_argument(
        "--permutation",
        default=False,
        action="store_true",
        help="Equal Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--randompermutation",
        default=False,
        action="store_true",
        help="Random Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--magwarp",
        default=False,
        action="store_true",
        help="Magnitude warp preset augmentation",
    )
    parser.add_argument(
        "--timewarp",
        default=False,
        action="store_true",
        help="Time warp preset augmentation",
    )
    parser.add_argument(
        "--windowslice",
        default=False,
        action="store_true",
        help="Window slice preset augmentation",
    )
    parser.add_argument(
        "--windowwarp",
        default=False,
        action="store_true",
        help="Window warp preset augmentation",
    )
    parser.add_argument(
        "--rotation",
        default=False,
        action="store_true",
        help="Rotation preset augmentation",
    )
    parser.add_argument(
        "--spawner",
        default=False,
        action="store_true",
        help="SPAWNER preset augmentation",
    )
    parser.add_argument(
        "--dtwwarp",
        default=False,
        action="store_true",
        help="DTW warp preset augmentation",
    )
    parser.add_argument(
        "--shapedtwwarp",
        default=False,
        action="store_true",
        help="Shape DTW warp preset augmentation",
    )
    parser.add_argument(
        "--wdba",
        default=False,
        action="store_true",
        help="Weighted DBA preset augmentation",
    )
    parser.add_argument(
        "--discdtw",
        default=False,
        action="store_true",
        help="Discrimitive DTW warp preset augmentation",
    )
    parser.add_argument(
        "--discsdtw",
        default=False,
        action="store_true",
        help="Discrimitive shapeDTW warp preset augmentation",
    )
    parser.add_argument("--extra_tag", type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")

    # FACT (ICLR 2026), https://github.com/wanghq21/FACT
    parser.add_argument(
        "--dilation",
        type=int,
        nargs="+",
        default=[1, 2, 3, 2, 1],
        help="FACT: dilation per Inception block; its length is the number of blocks",
    )
    parser.add_argument(
        "--core",
        type=float,
        default=0.5,
        help="FACT: weight between time-domain and frequency-domain modelling",
    )

    # MixLinear (ICLR 2026), https://github.com/aitianma/MixLinear
    parser.add_argument("--period_len", type=int, default=24, help="MixLinear: period length")
    parser.add_argument("--lpf", type=int, default=15, help="MixLinear: low-pass cutoff")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="MixLinear: time/frequency mix factor"
    )
    # RAG
    parser.add_argument(
        "--fusion_mode", help="fusion mode of RAG, mean or mlp", default="mean"
    )
    parser.add_argument("--w_trend", type=float, help="weight of trend", default=0.25)
    parser.add_argument(
        "--w_frequency", type=float, help="weight of frequency", default=0.25
    )
    parser.add_argument("--use_rag", action="store_true", help="whether use rag", default=False)
    parser.add_argument(
        "--num_retrieve", type=int, help="number of retrieve (k)", default=5
    )
    parser.add_argument("--gamma_1", type=float, default=0.5, help="weight for base loss")
    parser.add_argument("--gamma_2", type=float, default=0.5, help="weight for RL loss")
    parser.add_argument("--num_rl_samples", type=int, default=8, help="number of RL samples for policy gradient")

    parser.add_argument(
        "--gamma_3",
        type=float,
        default=0.0,
        help="weight of the corrector->backbone distillation term. 0 = the "
        "submitted method (RL term only, transfer left to zero-order channels). "
        ">0 gives theta the corrector's accepted output as an explicit target",
    )
    parser.add_argument(
        "--distill_target",
        type=str,
        default="best",
        choices=["best", "advantage"],
        help="which sampled correction to distil: the highest-reward one per "
        "timestep, or a reward-softmax weighted mean",
    )
    parser.add_argument("--distill_tau", type=float, default=1.0, help="softmax temperature")
    parser.add_argument(
        "--distill_all",
        action="store_true",
        default=False,
        help="distil every correction, including those that improved nothing "
        "(default: only corrections with reward > 0)",
    )

    # CRAFT corrector details (all previously hardcoded or missing)
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.0,
        help="weight of the L2 action penalty in L_RL (paper Eq. 12). 0 reproduces "
        "the released implementation, which omitted this term",
    )
    parser.add_argument("--kappa", type=int, default=3, help="temporal pooling window")
    parser.add_argument(
        "--reward_level",
        type=str,
        default="step",
        choices=["step", "item"],
        help="granularity of the reward: per forecast timestep or per sample",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
        help="discrete {0,1,2} indicators (paper) or raw improvement magnitude",
    )
    parser.add_argument(
        "--rl_sampling",
        type=str,
        default="sample",
        choices=["sample", "rsample"],
        help="'sample' = REINFORCE score-function estimator (paper); 'rsample' = "
        "reparameterised, kept as a sensitivity knob",
    )
    parser.add_argument(
        "--detach_yhat",
        action="store_true",
        default=False,
        help="block the RL gradient path into the backbone (detach ablation)",
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="nn",
        choices=["nn", "random"],
        help="nearest-neighbour retrieval, or the random-reference control",
    )
    parser.add_argument(
        "--exclusion_radius",
        type=int,
        default=0,
        help="drop neighbours within this many timesteps of the query. 0=off, "
        "1=self-exclusion, >=pred_len = no target-window overlap, "
        ">=seq_len+pred_len = no overlap at all",
    )
    parser.add_argument(
        "--freeze_policy",
        action="store_true",
        default=False,
        help="keep the policy head at its random initialisation (never updated). "
        "This reproduces the configuration that actually produced the submitted "
        "numbers, since the old optimizer never stepped the policy head. As a "
        "deliberate design it makes the corrector a fixed random critic whose RL "
        "term acts as a structured regulariser on theta",
    )
    parser.add_argument("--policy_hidden", type=int, default=128, help="policy hidden width")
    parser.add_argument(
        "--policy_mode", type=str, default="concat", choices=["concat", "diff"]
    )
    parser.add_argument(
        "--memory_store_cpu",
        action="store_true",
        default=False,
        help="keep the retrieved-future store in host memory (needed for Traffic-scale data)",
    )
    parser.add_argument(
        "--no_faiss_gpu",
        action="store_true",
        default=False,
        help="force a CPU FAISS index even when CUDA is available",
    )
    parser.add_argument("--tag", type=str, default="", help="free-form suffix for the run id")
    parser.add_argument(
        "--no_save_arrays",
        action="store_true",
        default=False,
        help="skip writing pred.npy/true.npy (~160 MB per Weather run). Metrics and "
        "per-sample errors are still written. Use for large campaigns; the full "
        "arrays are only needed by experiments/freq_band_analysis.py",
    )
    parser.add_argument(
        "--slim_ckpt",
        action="store_true",
        default=False,
        help="delete the full wrapper checkpoint after extracting base_model.pth, "
        "keeping only the deployable backbone weights",
    )
    parser.add_argument(
        "--skip_if_done",
        action="store_true",
        default=False,
        help="exit immediately if runs/<setting>.json already exists. Makes a whole "
        "campaign resumable: re-launching only runs what is missing",
    )
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        default=False,
        help="enable torch.autograd.set_detect_anomaly (debugging only; slow)",
    )

    args = parser.parse_args()

    # Seeding: previously fixed at 2021 regardless of --seed, which made
    # multi-seed evaluation impossible from the CLI.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.distill_only_positive = not args.distill_all
    args.save_arrays = not args.no_save_arrays

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device("cuda:{}".format(args.gpu))
        print("Using GPU")
    else:
        # Keep use_gpu/gpu_type consistent with reality, otherwise Exp_Basic
        # still tries to move the model to cuda:0 and crashes on CPU-only hosts.
        args.use_gpu = False
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
            args.use_gpu = True
            args.gpu_type = "mps"
        else:
            args.device = torch.device("cpu")
        print(f"Using {args.device}")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = build_setting(args, ii)
            if args.skip_if_done and os.path.exists(
                os.path.join("runs", f"{setting}.json")
            ):
                print(f">>>>>>>skip (already done) : {setting}")
                continue

            exp = Exp(args)  # set experiments

            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)
            if args.gpu_type == "mps":
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda":
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = build_setting(args, ii)

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == "mps":
            torch.backends.mps.empty_cache()
        elif args.gpu_type == "cuda":
            torch.cuda.empty_cache()
