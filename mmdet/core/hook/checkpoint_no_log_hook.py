from mmcv.runner.dist_utils import allreduce_params

from mmcv.runner.hooks import CheckpointHook, HOOKS


@HOOKS.register_module()
class CheckpointNoLogHook(CheckpointHook):
    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(runner, self.interval) or (self.save_last and self.is_last_epoch(runner)):
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(runner, self.interval) or (self.save_last and self.is_last_iter(runner)):
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)
