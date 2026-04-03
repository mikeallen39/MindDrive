_base_ = ['./minddrive_qwen2_05B_infer.py']

ida_aug_conf = dict(
    resize_lim=(0.37, 0.45),
    final_dim=(320, 640),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=704,
    W=1280,
    rand_flip=False,
)

model = dict(
    save_path='./results_latency_1280x704/',
)
