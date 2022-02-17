# dataset settings
dataset_type = 'DualCervixDataSet'
data_root = 'data/DualCervixDetection/'
modal = 'iodine'
img_norm_cfg = dict(
    acid = dict(
        mean = [107.5614013671875, 108.2771987915039, 157.57847595214844],
        std = [15.73247241973877, 13.8170747756958, 24.39257049560547],
        to_rgb = True),
    iodine = dict(
        mean = [43.90464401245117, 71.49726867675781, 123.6583023071289],
        std = [14.21172046661377, 18.04821014404297, 34.25565719604492],
        to_rgb = True))
img_scale = (512, 512)
train_pipeline = dict(
    acid = [
        dict(type = 'LoadImageFromFile'),
        dict(type = 'LoadAnnotations', with_bbox = True),
        dict(type = 'Resize', img_scale = img_scale, keep_ratio = True),
        dict(type = 'RandomFlip', flip_ratio = 0.5),
        dict(type = 'Normalize', **img_norm_cfg['acid']),
        dict(type = 'Pad', size_divisor = 32),
        dict(type = 'DefaultFormatBundle'),
        dict(type = 'Collect', keys = ['img', 'gt_bboxes', 'gt_labels']), ],
    iodine = [
        dict(type = 'LoadImageFromFile'),
        dict(type = 'LoadAnnotations', with_bbox = True),
        dict(type = 'Resize', img_scale = img_scale, keep_ratio = True),
        dict(type = 'RandomFlip', flip_ratio = 0.5),
        dict(type = 'Normalize', **img_norm_cfg['iodine']),
        dict(type = 'Pad', size_divisor = 32),
        dict(type = 'DefaultFormatBundle'),
        dict(type = 'Collect', keys = ['img', 'gt_bboxes', 'gt_labels']), ])
test_pipeline = dict(
    acid = [
        dict(type = 'LoadImageFromFile'),
        dict(
            type = 'MultiScaleFlipAug',
            img_scale = img_scale,
            flip = False,
            transforms = [
                dict(type = 'Resize', keep_ratio = True),
                dict(type = 'RandomFlip'),
                dict(type = 'Normalize', **img_norm_cfg['acid']),
                dict(type = 'Pad', size_divisor = 32),
                dict(type = 'ImageToTensor', keys = ['img']),
                dict(type = 'Collect', keys = ['img'])])],
    iodine = [
        dict(type = 'LoadImageFromFile'),
        dict(
            type = 'MultiScaleFlipAug',
            img_scale = img_scale,
            flip = False,
            transforms = [
                dict(type = 'Resize', keep_ratio = True),
                dict(type = 'RandomFlip'),
                dict(type = 'Normalize', **img_norm_cfg['iodine']),
                dict(type = 'Pad', size_divisor = 32),
                dict(type = 'ImageToTensor', keys = ['img']),
                dict(type = 'Collect', keys = ['img'])])])
data = dict(
    samples_per_gpu = 8,
    workers_per_gpu = 8,
    train = dict(
        type = dataset_type,
        modal = modal,
        ann_file = data_root + 'cropped_annos/train_{modal}.json',
        img_prefix = data_root + 'cropped_img/',
        pipeline = train_pipeline),
    val = dict(
        type = dataset_type,
        modal = modal,
        ann_file = data_root + 'cropped_annos/val_{modal}.json',
        img_prefix = data_root + 'cropped_img/',
        pipeline = test_pipeline),
    test = dict(
        type = dataset_type,
        modal = modal,
        ann_file = data_root + 'cropped_annos/test_{modal}.json',
        img_prefix = data_root + 'cropped_img/',
        pipeline = test_pipeline))
evaluation = dict(interval = 1, metric = 'bbox')
