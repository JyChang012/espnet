from collections import defaultdict
from typing import Dict, Callable, List, Mapping, Any, Sequence

import jax
import numpy as np
from jax import tree_util
from flax.traverse_util import flatten_dict


def calculate_all_attentions(
        get_attention_weight_step: Callable, params: Mapping[str, Any], batch: Dict[str, np.ndarray]
) -> Dict[str, List[np.ndarray]]:
    bs = len(next(iter(batch.values())))
    assert all(len(v) == bs for v in batch.values()), 'batch sizes are not the same: ' + str({
        k: v.shape for k, v in batch.items()
    })

    # forward the sample one by one
    keys = []
    for k in batch:
        if not (k.endswith("_lengths") or k in ["utt_id"]):
            keys.append(k)

    return_dict = defaultdict(list)

    for ibatch in range(bs):
        # *: (B, L, ...) -> (1, L2, ...)
        _sample = {
            k: batch[k][ibatch, None, : batch[k + "_lengths"][ibatch]]
            if k + "_lengths" in batch
            else batch[k][ibatch, None]
            for k in keys
        }

        # *_lengths: (B,) -> (1,)
        _sample.update(
            {
                k + "_lengths": batch[k + "_lengths"][ibatch, None]
                for k in keys
                if k + "_lengths" in batch
            }
        )

        if "utt_id" in batch:
            _sample["utt_id"] = batch["utt_id"]

        outputs = get_attention_weight_step(params, _sample)
        outputs = jax.device_get(outputs)

        for name, output in outputs.items():
            # TODO: support more attention type!
            # output: (1, NHead, Tout, Tin) -> (NHead, Tout, Tin)
            output = output.squeeze(0)
            # output: (Tout, Tin) or (NHead, Tout, Tin)
            return_dict[name].append(output)
    return_dict.default_factory = None
    return return_dict
