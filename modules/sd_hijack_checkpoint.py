from torch.utils.checkpoint import checkpoint
from modules.shared import opts

import ldm.modules.attention
import ldm.modules.diffusionmodules.openaimodel


def BasicTransformerBlock_forward(self, x, context=None):
    # CLIP guidance does not support checkpointing due to the use of torch.autograd.grad()
    if opts.clip_guidance:
        return self._forward(x, context)
    else:
        return checkpoint(self._forward, x, context)


def AttentionBlock_forward(self, x):
    if opts.clip_guidance:
        return self._forward(x)
    else:
        return checkpoint(self._forward, x)


def ResBlock_forward(self, x, emb):
    if opts.clip_guidance:
        return self._forward(x, emb)
    else:
        return checkpoint(self._forward, x, emb)

stored = []


def add():
    if len(stored) != 0:
        return

    stored.extend([
        ldm.modules.attention.BasicTransformerBlock.forward,
        ldm.modules.diffusionmodules.openaimodel.ResBlock.forward,
        ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward
    ])

    ldm.modules.attention.BasicTransformerBlock.forward = BasicTransformerBlock_forward
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = ResBlock_forward
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = AttentionBlock_forward


def remove():
    if len(stored) == 0:
        return

    ldm.modules.attention.BasicTransformerBlock.forward = stored[0]
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = stored[1]
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = stored[2]

    stored.clear()

