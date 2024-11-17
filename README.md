# 一个关于训练glm4-4v-9b的仓库 (尝试用2gpu进行fb16的平民化训练)
存在的问题:
    [rank1]: torch.utils.checkpoint.CheckpointError: torch.utils.checkpoint: Recomputed values for the following tensors have different metadata than during the forward pass.
    [rank1]: tensor at position 4:
    [rank1]: saved metadata: {'shape': torch.Size([4096]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: recomputed metadata: {'shape': torch.Size([0]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: tensor at position 6:
    [rank1]: saved metadata: {'shape': torch.Size([4608, 4096]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: recomputed metadata: {'shape': torch.Size([0]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: tensor at position 7:
    [rank1]: saved metadata: {'shape': torch.Size([4608]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: recomputed metadata: {'shape': torch.Size([0]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: tensor at position 29:
    [rank1]: saved metadata: {'shape': torch.Size([4096, 4096]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: recomputed metadata: {'shape': torch.Size([0]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: tensor at position 34:
    [rank1]: saved metadata: {'shape': torch.Size([4096]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: recomputed metadata: {'shape': torch.Size([0]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: tensor at position 36:
    [rank1]: saved metadata: {'shape': torch.Size([27392, 4096]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
    [rank1]: recomputed metadata: {'shape': torch.Size([0]), 'dtype': torch.float32, 'device': device(type='cuda', index=1)}
问题所在不清晰且在autodl可正常训练 当前已知存在几个相同的 且都已尝为解决推测与 deepspeed有关