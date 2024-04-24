#.............................................................................................................................................................. Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np

from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH
from ..utils.ops import xywh2ltwh
from ..utils import LOGGER


class STrack(BaseTrack):
    """
    单个目标跟踪的表示类，它使用卡尔曼滤波（Kalman Filter）进行状态估计。
    Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): 共享的 Kalman 滤波器 Shared Kalman filter that is used across all STrack instances for prediction.
        _tlwh (np.ndarray): 一个私有属性，存储目标边界框的左上角坐标、宽度和高度Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): 用于此对象跟踪的 Kalman 滤波器实例Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): 状态估计向量的均值Mean state estimate vector.
        covariance (np.ndarray):状态估计的协方差。Covariance of state estimate.
        is_activated (bool):布尔标志，指示跟踪是否已激活。 Boolean flag indicating if the track has been activated.
        score (float): 跟踪的置信度分数。Confidence score of the track.
        tracklet_len (int): 跟踪长度 表示跟踪目标已经持续的帧数。Length of the tracklet.
        cls (any):对象的类别标签。 Class label for the object.
        idx (int): 对象的索引或标识符。Index or identifier for the object.
        frame_id (int): 当前帧的 ID。Current frame ID.
        start_frame (int): 首次检测到对象的帧。Frame where the object was first detected.

    Methods:
        predict(): 使用 Kalman 滤波器预测对象的下一个状态 Predict the next state of the object using Kalman filter.
        multi_predict(stracks): 对给定的多个 STrack 实例执行多对象预测跟踪。Predict the next states for multiple tracks.
        multi_gmc(stracks, H):使用单应性矩阵更新多个跟踪的状态。 Update multiple track states using a homography matrix.
        activate(kalman_filter, frame_id): 激活一个新的跟踪。Activate a new tracklet.
        re_activate(new_track, frame_id, new_id):使用新检测重新激活先前丢失的跟踪。 Reactivate a previously lost tracklet.
        update(new_track, frame_id):更新匹配跟踪的状态。 Update the state of a matched track.
        convert_coords(tlwh):  将边界框的格式从左上角-宽度-高度转换为中心点-宽度-高度-角度。Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah(tlwh): 将边界框的格式从左上角-宽度-高度转换为中心点-宽度-高度-角度。Convert tlwh bounding box to xyah format.
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        """Initialize new STrack instance."""
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in [5, 6], f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

        #!!!!!
        self.keep_tracked = 0
        self.keep_lost = 0
        self.lost = 0
        #!!!!!

    def predict(self):
        """用于使用 Kalman 滤波器进行状态预测。 Predicts mean and covariance using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0 # 将速度分量的均值设为零来抑制预测的速度，以便更好地处理丢失的跟踪轨迹。
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for given stracks."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """更新一组目标跟踪的位置和协方差，以适应不同视角或平面的目标跟踪需求。
        通常用于多摄像头系统中，用于处理不同摄像头之间的视角变换或者透视变换。
    Update state tracks positions and covariances using a homography matrix."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """启动一个新的跟踪轨迹。 Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        print("frame_id:", frame_id)
        if frame_id == 1:   ######！！！！！！ 为什么要判断来激活
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """重新激活先前丢失的跟踪轨迹 Reactivates a previously lost track with a new detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track, frame_id):
        """
        更新已匹配跟踪对象的状态。修正卡尔曼滤波器（均置和协方差）
        Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Get current position in bounding box format (top left x, top left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        """Convert bounding box to format (min x, min y, max x, max y), i.e., (top left, bottom right)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format (center x, center y, aspect ratio, height), where the aspect ratio is width /
        height.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        """Get current position in bounding box format (center x, center y, width, height)."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        """Get current position in bounding box format (center x, center y, width, height, angle)."""
        if self.angle is None:
            LOGGER.warning("WARNING ⚠️ `angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        """Get current tracking results."""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):
        """Return a string representation of the BYTETracker object with start and end frames and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """
    一个基于 YOLOv8 的目标检测和跟踪算法。它负责初始化、更新和管理视频序列中检测到的对象的跟踪轨迹。以下是这个类的一些重要属性和方法：
    BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    The class is responsible for initializing, updating, and managing the tracks for detected objects in a video
    sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for
    predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks (list[STrack]): 成功激活的跟踪轨迹列表。 List of successfully activated tracks.
        lost_stracks (list[STrack]): 丢失的跟踪轨迹列表。 List of lost tracks.
        removed_stracks (list[STrack]): 移除的跟踪轨迹列表。 List of removed tracks.
        frame_id (int): 当前帧的 ID。 The current frame ID.
        args (namespace): 命令行参数。 Command-line arguments.
        max_time_lost (int): 被认为是“丢失”的跟踪轨迹的最大帧数。 The maximum frames for a track to be considered as 'lost'.
        kalman_filter (object): 用于跟踪边界框的卡尔曼滤波器对象。 Kalman Filter object.

    Methods:
        update(results, img=None):使用新的检测结果更新对象跟踪器，并返回跟踪的对象边界框。 Updates object tracker with new detections.
        get_kalmanfilter():返回一个用于跟踪边界框的卡尔曼滤波器对象。 Returns a Kalman filter object for tracking bounding boxes.
        init_track(dets, scores, cls, img=None): 使用检测和分数初始化对象跟踪。Initialize object tracking with detections.
        get_dists(tracks, detections): 计算跟踪和检测之间的距离，使用 IOU 并融合分数。Calculates the distance between tracks and detections.
        multi_predict(tracks):使用 YOLOv8 网络返回预测的跟踪。 Predicts the location of tracks.
        reset_id(): 重置 STrack 的 ID 计数器。Resets the ID counter of STrack.
        joint_stracks(tlista, tlistb): 将两个跟踪列表合并成一个。Combines two lists of stracks.
        sub_stracks(tlista, tlistb): 从第一个列表中筛选出在第二个列表中不存在的跟踪。Filters out the stracks present in the second list from the first list.
        remove_duplicate_stracks(stracksa, stracksb): ：删除重复的跟踪轨迹，使用非最大 IOU 距离。Removes duplicate stracks based on IOU.
    """

    def __init__(self, args, frame_rate=30):
        """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        ### !!!!!
        self.stracks = []  # type: list[STrack]
        self.track_id = -1
        ### !!!!!

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
        self.debug = True

    def sort(self):
        self.stracks = sorted(self.stracks, key=lambda x: x.lost, reverse=True)

    def match(self, track, detections):
        dists = matching.iou_distance(track, detections)
        return dists


    def update(self, results, img=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        print("new frame")
        self.frame_id += 1
        stracks = []

        # 从检测结果中获取置信度、边界框和类别信息。
        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls
        if self.debug:
            print(f"frame_id:{self.frame_id}")
            print(f"bboxes:{bboxes} cls:{cls}")
            print(f"scores:{scores}")

        # 根据阈值对置信度进行筛选，得到保留的高置信度检测框和低置信度检测框。
        high_thresh_index = scores > self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        low_thresh_index = np.logical_and(inds_low, inds_high)

        low_thresh_boxes = bboxes[low_thresh_index]        # 低置信度观测边框
        high_thresh_boxes = bboxes[high_thresh_index]      # 高置信度观测边框
        high_thresh_score = scores[high_thresh_index]      # 高置信度
        low_thresh_score = scores[low_thresh_index]        # 低置信度
        high_thresh_class = cls[high_thresh_index]         # 高置信度类别
        low_thresh_class = cls[low_thresh_index]           # 低置信度类别

        if self.debug:
            print("low_thresh_boxes:", low_thresh_boxes)
            print("high_thresh_boxes:", high_thresh_boxes)
            print("high_thresh_score:", high_thresh_score)
            print("low_thresh_score:", low_thresh_score)
            print("high_thresh_class:", high_thresh_class)
            print("low_thresh_class:", low_thresh_class)

        # 将高置信度检测框初始化为轨迹对象。
        detections = self.init_track(high_thresh_boxes, high_thresh_score, high_thresh_class, img)
        print(f"lendetections:{len(detections)}")
        print(f"lentrackers:{len(self.stracks)}")

        if self.frame_id != 1:

            # 将当前的追踪器自行排序
            self.sort()
            # 对当前追踪器进行预测
            self.multi_predict(self.stracks)

            print(f"end_multi_predict")

            # 将追踪器依次和检测框进行匹配
            matched_tracks = []
            tracks = []
            for track in self.stracks:
                tracks.clear()
                tracks.append(track)
                dists = matching.iou_distance2(tracks, detections)
                # print(f"high_dists:{dists}")
                i = 0
                for dist in dists[0, :]:
                    print(f"high_dist:{dist}")
                    if len(detections) == 0:
                        # print("get no img")
                        break
                    if (len(dists) == 1 and dist < 1) or (dist < 0.95 and (dist + 0.05) < np.min(np.delete(dists, i))):
                        print("get high match")
                        # self.multi_predict(tracks)
                        track.update(detections[i], self.frame_id)
                        detections.pop(i)
                        matched_tracks.append(track)
                        break
                    i += 1
            unmatched_tracks = list(set(self.stracks) - set(matched_tracks))

            print(f"end_high_match")

            # 将低置信度的检测框和与未匹配成功的追踪器相匹配
            detections_low = self.init_track(low_thresh_boxes, low_thresh_score, low_thresh_class, img)
            # 匹配方式与上面的一致
            matched_tracks_low = []
            tracks2 = []
            for track in unmatched_tracks:
                tracks2.clear()
                tracks2.append(track)
                dists = matching.iou_distance2(tracks2, detections_low)
                # print(f"low_dists:{dists}")
                i = 0
                for dist in dists[0, :]:
                    print(f"low_dist:{dist}")
                    if len(detections_low) == 0:
                        # print("get no img")
                        break
                    if (len(dists) == 1 and dist < 1) or (dist < 0.95 and (dist + 0.05) < np.min(np.delete(dists, i))):
                        print("get low match")
                        # self.multi_predict(tracks)
                        track.update(detections_low[i], self.frame_id)
                        detections_low.pop(i)
                        matched_tracks_low.append(track)
                        break
                    i += 1
            unmatched_tracks_low = list(set(unmatched_tracks) - set(matched_tracks_low))

            print(f"end_low_match")

            # 匹配成功的放到一起
            for track_list in [matched_tracks, matched_tracks_low]:
                for track in track_list:
                    track.keep_lost = 0
                    track.keep_tracked += 1
                    stracks.append(track)
                    print(f"matched_track_id:{track.track_id}")


            # 如果track_id匹配成功，则不更新，否则进行更新
            i = 0
            print(f"track id match strackslen:{len(stracks)}")
            for track in stracks:
                if track.track_id == self.track_id:
                    print("get track id")
                    stracks.insert(0, stracks.pop(i))
                    break
                i += 1

            # 清空追踪器并将当前匹配成功的追踪器放进去
            self.stracks.clear()
            for track in stracks:
                self.stracks.append(track)

            # 依然没有匹配成功的track标记为lost++,若lost大于阈值则删除该追踪器
            for track in unmatched_tracks_low:
                track.keep_lost += 1
                track.keep_tracked = 0
                if track.keep_lost < 10:
                    self.stracks.append(track)
                else:
                    print(f"track{track.track_id} has been removed")

            # 高置信度检测框初始化为轨迹
            for track in detections:
                track.activate(self.kalman_filter, self.frame_id)
                track.keep_lost = 0
                track.keep_tracked = track.keep_tracked + 1
                stracks.append(track)
                self.stracks.append(track)
        else:
            print(f"frame1: lendetections:{len(detections)}")
            if len(detections) > 0:
                for track in detections:
                    track.activate(self.kalman_filter, self.frame_id)
                    track.keep_lost = 0
                    track.keep_tracked = track.keep_tracked + 1
                    stracks.append(track)
                self.track_id = 1
                self.stracks = stracks
            else:
                print("no detection, frame_id --")
                self.frame_id = self.frame_id - 1

        # 计算 lost 值
        for track in self.stracks:
            track.lost = track.keep_tracked - track.keep_lost

        self.track_id = stracks[0].track_id if len(stracks) > 0 else self.track_id
        print(f"len(stracks):{len(stracks)}")
        print(f"len_selfstracks:{len(self.stracks)}")
        print(f"track_id:{self.track_id}")
        for track in self.stracks:
            print(f"self_track_id:{track.track_id}")
        return np.asarray([stracks[0].result] if len(stracks) > 0 else [], dtype=np.float32)


    def update2(self, results, img=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 从检测结果中获取置信度、边界框和类别信息。
        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        # 根据阈值对置信度进行筛选，得到保留的高置信度检测框和低置信度检测框。
        # 低置信度指在track_low_thresh和track_high_thresh之间的置信度
        remain_inds = scores > self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second] # 低置信度观测边框
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        # 将高置信度检测框初始化为轨迹对象。
        detections = self.init_track(dets, scores_keep, cls_keep, img)
        # Add newly detected tracklets to tracked_stracks
        # 将未激活的轨迹加入 unconfirmed 列表，已激活的轨迹加入 tracked_stracks 列表。
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        # Step 2: First association, with high score detection boxes
        # 将已激活的和丢失的轨迹池合并。
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # for track in self.tracked_stracks:
        #     if not track.is_activated:
        #         print("tracked_stracks is_activate:", track.track_id)
        #
        # for track in self.lost_stracks:
        #     if not track.is_activated:
        #         print("lost_stracks is_activate:", track.track_id)
        #
        # for track in self.removed_stracks:
        #     if not track.is_activated:
        #         print("removed_stracks is_activate:", track.track_id)


        # 使用卡尔曼滤波器预测轨迹池中各个轨迹的当前位置。（tracked+lost）
        self.multi_predict(strack_pool)

        # 如果存在图像变换器 (`gmc`) 并且输入了图像，则对检测框进行几何变换
        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # 计算轨迹池中各个轨迹与高置信度检测框之间的距离。
        dists = self.get_dists(strack_pool, detections)
        print(f"dists:{dists}")
        # dists2 = matching.iou_distance(strack_pool, detections)

        # 输出距离值和置信度
        # for i, dist in enumerate(dists):
        #     print(f"ALL track {i} and detections: {dist}")
        # for i, dist in enumerate(dists2):
        #     print(f"IOU track {i} and detections: {dist}")
        # 使用匈牙利算法进行关联匹配。
        # matches是一个列表，包含了成功匹配的跟踪目标和检测结果的索引对。
        # u_track是一个列表，包含了未匹配的跟踪目标的索引。
        # u_detection是一个列表，包含了未匹配的检测结果的索引。
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # 根据匹配结果，更新已跟踪的轨迹状态或重新激活丢失的轨迹
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections

        # 将低置信度的检测框和与未匹配的追踪器相匹配，这里只做IOU_match
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # TODO
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        # 根据匹配结果，更新已跟踪的轨迹状态或重新激活丢失的轨迹
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 没有匹配成功的track标记为Lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        # 处理未激活的轨迹，通常是只出现了一帧的轨迹，还未与其他帧目标相关联，这里就让他们尝试和落选的检测目标相关联
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:  # 未激活的没匹配上就Remove
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        # Step 4: Init new stracks
        for inew in u_detection: # 没有匹配上的目标高于阈值则新建一个track
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        # Step 5: Update state
        # lost超过一定时间则remove
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum

        # print("frame_id:", self.frame_id)
        # for strack in self.tracked_stracks:
        #     print("Data in tracked_strack:")
        #     print("Track ID:", strack.track_id)
        #     print("Other state:", strack.state)
        #     print("Other score:", strack.score)

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes."""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracking with detections and scores using STrack algorithm."""
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IOU and fuses scores."""
        dists = matching.iou_distance(tracks, detections)
        # TODO: mot20
        # if not self.args.mot20:
        # dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """Returns the predicted tracks using the YOLOv8 network."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Resets the ID counter of STrack."""
        STrack.reset_id()

    def reset(self):
        """Reset tracker."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combine two lists of stracks into a single one."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """DEPRECATED CODE in https://github.com/ultralytics/ultralytics/pull/1890/
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())
        """
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Remove duplicate stracks with non-maximum IOU distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
