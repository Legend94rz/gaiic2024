from mmengine import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


if __name__ == "__main__":
    register_all_modules()
    cfg = Config.fromfile("/home/renzhen/userdata/repo/gaiic2024/projects/gaiic2014/configs/codetr_all_in_one.py")
    model = MODELS.build(cfg['model'])
    print(model)
