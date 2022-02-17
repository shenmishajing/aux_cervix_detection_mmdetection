import re

from mmcv.runner import master_only
from mmcv.runner.hooks import HOOKS, WandbLoggerHook


@HOOKS.register_module()
class WandbLoggerConfigHook(WandbLoggerHook):
    """WandbLoggerHook with logging code and save config feature.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self,
                 root = '.',
                 name = None,
                 include_pattern = None,
                 exclude_pattern = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.name = name

        if include_pattern is None:
            include_pattern = ['py', 'yaml', 'yml', 'json', 'sh', 'md', 'txt']
        elif isinstance(include_pattern, str):
            include_pattern = [include_pattern]
        include_pattern = [re.compile(rf'.*\.{p}$') for p in include_pattern]

        if exclude_pattern is None:
            exclude_pattern = ['wandb', 'venv', 'temp', 'tmp', 'work_dirs', 'requirements', r'\.', r'.*\.egg-info', r'setup.py']
        elif isinstance(exclude_pattern, str):
            exclude_pattern = [exclude_pattern]
        ex_p = []
        for p in exclude_pattern:
            ex_p.extend([re.compile(rf'.*/{p}.*'), re.compile(rf'^[/]{p}.*')])

        self.include_fn = lambda path: any([p.fullmatch(path) for p in include_pattern])
        self.exclude_fn = lambda path: any([p.fullmatch(path) for p in ex_p])

    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        # log config
        self.wandb.config.update(runner.meta['config'])

        # log code
        if self.name is None:
            self.name = self.wandb.run.project_name()
        self.wandb.run.log_code(root = self.root, name = self.name, include_fn = self.include_fn, exclude_fn = self.exclude_fn)

        # watch model
        self.wandb.watch(runner.model)
