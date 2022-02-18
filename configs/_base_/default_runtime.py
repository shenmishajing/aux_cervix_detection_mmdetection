checkpoint_config = dict(type = 'CheckpointNoLogHook', interval = 1)
# yapf:disable
log_config = dict(
    interval = 50,
    hooks = [
        dict(type = 'WandbLoggerConfigHook',
             with_step = False,
             init_kwargs = dict(project = 'aux_cervix_detection')),
        dict(type = 'RichProgressBarHook')])
# yapf:enable
custom_hooks = [dict(type = 'NumClassCheckHook')]

dist_params = dict(backend = 'nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
