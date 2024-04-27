from mmengine.hooks import EMAHook
import copy
import logging
from mmengine.logging import print_log
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MyEMAHook(EMAHook):
    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """Resume ema parameters from checkpoint.

        Args:
            runner (Runner): The runner of the testing process.
        """
        from mmengine.runner.checkpoint import load_state_dict
        if 'ema_state_dict' in checkpoint and runner._resume:
            # The original model parameters are actually saved in ema
            # field swap the weights back to resume ema state.
            self._swap_ema_state_dict(checkpoint)
            self.ema_model.load_state_dict(
                checkpoint['ema_state_dict'], strict=self.strict_load)

        # Support load checkpoint without ema state dict.
        else:
            if runner._resume:
                print_log(
                    'There is no `ema_state_dict` in checkpoint. '
                    '`EMAHook` will make a copy of `state_dict` as the '
                    'initial `ema_state_dict`', 'current', logging.WARNING)
            if 'state_dict' in checkpoint:
                load_state_dict(
                    self.ema_model.module,
                    copy.deepcopy(checkpoint['state_dict']),
                    strict=self.strict_load)
            else:
                print_log(f"Not found `state_dict` in the checkpoint, trying to load directly...", 'current', logging.WARNING)
                load_state_dict(self.ema_model.module, copy.deepcopy(checkpoint), strict=self.strict_load)
