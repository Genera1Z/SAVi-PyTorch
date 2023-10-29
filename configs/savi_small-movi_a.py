### datum

dataset_train = dict(
    type="MoviLmdb",
    data_file=...,
    transform=[
        dict(type="VideoFromTfds"),
        dict(type="SparseToDenseAnnotation", max_instances=23),
        dict(type="TemporalRandomStridedWindow", length=6),  # 24|6
        dict(type="ResizeSmall", size=64),  # 128|64
        dict(type="FlowToRgb"),
    ],
    max_spare=4,
)
dataset_val = dict(
    type="MoviLmdb",
    data_file=...,
    transform=[
        dict(type="VideoFromTfds"),
        dict(type="SparseToDenseAnnotation", max_instances=23),
        dict(type="TemporalCropOrPad", length=24),
        dict(type="ResizeSmall", size=64),  # 128|64
        dict(type="FlowToRgb"),
    ],
    max_spare=4,
)


### model

model = dict(
    type="Savi",
    enc_backbone=dict(
        type="SimpleCnn",
        in_channel=3,
        channels=[32, 32, 32, 32],
        kernels=[5, 5, 5, 5],
        strides=[1, 1, 1, 1],
        tranposes=[0, 0, 0, 0],
    ),
    enc_posit_embed=dict(type="PositionEmbedding2d", num_channel=32),
    enc_project=dict(
        type="MLP", in_channel=32, mid_channels=[64], out_channel=32, norm="pre"
    ),
    proc_initialize=dict(
        type="BboxEmbedding",
        mid_dim=256,
        out_dim=128,
        prepend_background=True,
        center_of_mass=False,
    ),
    proc_correct=dict(
        type="SlotAttention",
        num_iter=1,
        qi_dim=128,
        kvi_dim=32,
        qko_dim=128,
        vo_dim=128,
    ),
    proc_predict=dict(
        type="TransformBlock",
        embed_dim=128,
        num_head=4,
        q_dim=128,
        ffn_dim=256,
        pre_norm=False,
    ),
    dec_resolut=(8, 8),
    dec_posit_embed=dict(type="PositionEmbedding2d", num_channel=128),
    dec_backbone=dict(
        type="SimpleCnn",
        in_channel=128,
        channels=[64, 64, 64, 64],
        kernels=[5, 5, 5, 5],
        strides=[2, 2, 2, 1],
        tranposes=[1, 1, 1, 0],
    ),
    dec_readout=dict(type="Conv2d", in_channels=64, out_channels=4, kernel_size=1),
)
model_in_keys = dict(frames="video", condition="boxes")
model_out_keys = ["flow", "segment", "state", "logit"]

### learn

optimizer = dict(type="Adam", params=..., lr=2e-4)
sched_lr = dict(
    type="LambdaLR",
    optimizer=...,
    lr_lambda=dict(fn="linear_cosine", warmup_step=2500, total_step=...),
)

losses = dict(
    type="MetricDict",
    grad=True,
    mse=dict(fn="mse", weight=1.0, output=dict(pred="flow"), batch=dict(true="flow")),
)
metrics = dict(
    type="MetricDict",
    grad=False,
    ari=dict(
        fn="ari",
        weight=1.0,
        output=dict(idx_pd="segment"),
        batch=dict(idx_gt="segmentations", mask="padding_mask"),
        num_pd=24,
        num_gt=24,
        fg=False,
    ),
    ari_fg=dict(
        fn="ari",
        weight=1.0,
        output=dict(idx_pd="segment"),
        batch=dict(idx_gt="segmentations", mask="padding_mask"),
        num_pd=24,
        num_gt=24,
        fg=True,
    ),
)

num_epoch = 100
val_num_epoch = 5

batch_size_train = 16
batch_size_val = 4

num_worker_train = 4
num_worker_val = 4
