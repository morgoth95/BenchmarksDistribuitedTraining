from colossalai.amp import AMP_TYPE

BATCH_SIZE = 64
NUM_EPOCHS = 1
# NUM_MICRO_BATCHES=8

# Enable mixed precision training
fp16 = dict(
    mode=AMP_TYPE.TORCH
)
