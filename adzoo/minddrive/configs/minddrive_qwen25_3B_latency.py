REPO_ROOT = __import__('os').environ.get(
    'MINDDRIVE_ROOT',
    __import__('os').getcwd(),
)
LLM_PATH = __import__('os').environ.get(
    'MINDDRIVE_LLM_PATH',
    REPO_ROOT + '/ckpts/llava-qwen2.5-3b',
)

_base_ = ['./minddrive_qwen25_3B_infer.py']

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
    save_path='./results_latency_3b_1280x704/',
    tokenizer=LLM_PATH,
    lm_head=LLM_PATH,
)
