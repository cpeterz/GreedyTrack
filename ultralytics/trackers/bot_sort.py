# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import deque

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    """
     STrack 类的扩展，添加了一些用于目标跟踪的功能。
    An extended version of the STrack class for YOLOv8, adding object tracking features.

    Attributes:
        shared_kalman (KalmanFilterXYWH): 一个共享的 Kalman 滤波器 A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): 平滑后的特征向量。Smoothed feature vector.
        curr_feat (np.ndarray): 当前的特征向量Current feature vector.
        features (deque):一个双向队列，用于存储特征向量，其最大长度由 feat_history 定义。 A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): 用于特征平滑的指数移动平均的平滑因子。Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): Kalman 滤波器的均值状态。The mean state of the Kalman filter.
        covariance (np.ndarray): Kalman 滤波器的协方差矩阵。 The covariance matrix of the Kalman filter.

    Methods:
        update_features(feat):更新特征向量并使用指数移动平均进行平滑处理。Update features vector and smooth it using exponential moving average.
        predict(): 使用更新的特征重新激活跟踪，并可选地分配一个新的 ID。Predicts the mean and covariance using Kalman filter.
        re_activate(new_track, frame_id, new_id):: 使用新的跟踪和帧 ID 更新 YOLOv8 实例。 Reactivates a track with updated features and optionally new ID.
        update(new_track, frame_id): 使用新的跟踪和帧 ID 更新 YOLOv8 实例。Update the YOLOv8 instance with new track and frame ID.
        tlwh:获取当前位置的边界框格式  Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict(stracks):使用共享的 Kalman 滤波器预测多个目标跟踪的均值和协方差。 0Predicts the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords(tlwh):将左上角宽高形式的边界框坐标转换为中心宽高形式。 Converts tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh(tlwh):将边界框坐标转换为 (中心 x 坐标, 中心 y 坐标, 宽度, 高度) 格式。 Convert bounding box to xywh format `(center x, center y, width, height)`.

    Usage:
        bo_track = BOTrack(tlwh, score, cls, feat)
        bo_track.predict()
        bo_track.update(new_track, frame_id)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """Initialize YOLOv8 object with temporal parameters, such as feature history, alpha and current features."""
        super().__init__(tlwh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9 # 设置平滑处理特征向量的指数移动平均系数

    def update_features(self, feat):
        """用于更新特征向量并使用指数移动平均对其进行平滑处理。
         在更新目标的特征向量时，对其进行平滑处理以减少特征向量的抖动，并将其添加到特征历史记录中以便后续使用。Update features vector and smooth it using exponential moving average."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """使用卡尔曼滤波器预测轨迹的均值和协方差。Predicts the mean and covariance using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked: # 果当前轨迹状态不是被跟踪的状态（即不处于 Tracked 状态），则将均值的速度和加速度设置为零，这样可以防止在不跟踪的情况下预测出不准确的状态。
            mean_state[6] = 0
            mean_state[7] = 0

        # 使用卡尔曼滤波器的 predict 方法，传入当前状态的均值 mean_state 和协方差 covariance，得到预测后的均值和协方差。
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        重新激活（或更新）轨迹Reactivates a track with updated features and optionally assigns a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        """Update the YOLOv8 instance with new track and frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """Predicts the mean and covariance of multiple object tracks using shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        """Converts Top-Left-Width-Height bounding box coordinates to X-Y-Width-Height format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width, height)`."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """
    An extended version of the BYTETracker class for YOLOv8, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): 空间接近度阈值 用于确定跟踪和检测之间的匹配关系。Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): 外观相似度阈值 用于确定ReID（目标重新识别）嵌入的相似性。Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (object): 用于处理ReID嵌入的对象Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC):GMC算法的实例，用于数据关联。 An instance of the GMC algorithm for data association.
        args (object): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter():返回用于对象跟踪的KalmanFilterXYWH实例。 Returns an instance of KalmanFilterXYWH for object tracking.
        init_track(dets, scores, cls, img): 初始化具有检测、分数和类别的轨迹。如果启用了ReID，则使用嵌入来初始化轨迹。Initialize track with detections, scores, and classes.
        get_dists(tracks, detections):计算跟踪和检测之间的距离，包括IoU（交并比）和（可选的）ReID嵌入距离。 Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict(tracks): 使用共享的Kalman滤波器来预测和跟踪多个对象。Predict and track multiple objects with YOLOv8 model.

    Usage:
        bot_sort = BOTSORT(args, frame_rate)
        bot_sort.init_track(dets, scores, cls, img)
        bot_sort.multi_predict(tracks)

    Note:
        The class is designed to work with the YOLOv8 object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args, frame_rate=30):
        """Initialize YOLOv8 object with ReID module and GMC algorithm."""
        super().__init__(args, frame_rate)
        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            # Haven't supported BoT-SORT(reid) yet
            self.encoder = None

        if args.with_reid:
            print("ReID is enabled.")
            # Haven't supported BoT-SORT(reid) yet
            self.encoder = None
        else:
            print("ReID is not enabled.")
        self.gmc = GMC(method=args.gmc_method)
        if self.gmc.method:
            print(f"GMC is enabled with method: {self.gmc.method}.")
        else:
            print("GMC is not enabled.")
        self.gmc = GMC(method=args.gmc_method)

    def get_kalmanfilter(self):
        """Returns an instance of KalmanFilterXYWH for object tracking."""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """初始化跟踪对象，并基于检测结果、得分和类别创建相应的跟踪对象列表。Initialize track with detections, scores, and classes."""
        if len(dets) == 0:
            return []

        # 如果启用了ReID并且已经初始化了编码器（self.args.with_reid and self.encoder is not None），则通过编码器对图像和边界框进行推断，以获取特征向量。然后，使用检测边界框、得分、类别和特征向量创建BOTrack类的实例列表。
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder.inference(img, dets)
            return [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features_keep)]  # detections
        else:
            return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]  # detections

    def get_dists(self, tracks, detections):
        """Get distances between tracks and detections using IoU and (optionally) ReID embeddings."""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        # TODO: mot20
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks):
        """Predict and track multiple objects with YOLOv8 model."""
        BOTrack.multi_predict(tracks)

    def reset(self):
        """Reset tracker."""
        super().reset()
        self.gmc.reset_params()
