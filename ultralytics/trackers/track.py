# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial
from pathlib import Path

import torch

from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml
from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    在预测开始时初始化对象追踪器。
    Initialize trackers for object tracking during prediction.

    Args:
        predictor：表示要为其初始化追踪器的预测器对象。
        persist（可选）：一个布尔参数，指示是否在追踪器已存在时持久化追踪器。默认值为 False。
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
    """
    if hasattr(predictor, "trackers") and persist:  # 若已有追踪器且持久化，则无需重新初始化
        return

    # 从 predictor.args.tracker 中获取追踪器配置，并通过 check_yaml 函数对其进行检查。
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    if cfg.tracker_type not in ["bytetrack", "botsort"]:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    trackers = []
    # 对于每个预测器中的数据批次，根据配置中的 tracker_type 选择相应的追踪器类进行初始化，并将其添加到 trackers 列表中。
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
    predictor.trackers = trackers


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """
    bs = predictor.dataset.bs
    path, im0s = predictor.batch[:2]

    is_obb = predictor.args.task == "obb"
    for i in range(bs):
        # 如果不需要持久化追踪器且当前视频路径与上一帧不同，则重置追踪器，以处理新的视频。
        if not persist and predictor.vid_path[i] != str(predictor.save_dir / Path(path[i]).name):  # new video
            predictor.trackers[i].reset()

        # 获取当前帧的检测结果（det），如果没有检测到任何目标，则继续下一次迭代。

        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
        if len(det) == 0:
            continue

        # 使用追踪器更新检测结果，得到更新后的跟踪结果
        tracks = predictor.trackers[i].update(det, im0s[i])  # 更新
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)   # 从跟踪器的输出中提取了跟踪目标的索引
        predictor.results[i] = predictor.results[i][idx] # idx 是跟踪器返回的索引数组，它指示了当前帧中每个目标在前一帧中的索引。这里是在重新排序

        update_args = dict()
        update_args["obb" if is_obb else "boxes"] = torch.as_tensor(tracks[:, :-1])
        predictor.results[i].update(**update_args)




def register_tracker(model: object, persist: bool) -> None:
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
