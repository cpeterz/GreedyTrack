import os
import numpy as np

def load_labels(label_dir):
    labels = {}
    if "10_txt" in label_dir:
        for i in range(1, 1001):
            filename = f"{i:06d}.txt"
            filepath = os.path.join(label_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    labels[i] = [list(map(float, line.strip().split())) for line in lines]
            else:
                labels[i] = []
    else:
        for i in range(1, 1001):
            filename = f"output_video10_3_{i}.txt"
            filepath = os.path.join(label_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    if lines:
                        labels[i] = [list(map(float, line.strip().split())) for line in lines]
                    else:
                        labels[i] = []
            else:
                labels[i] = []
    return labels


def calculate(gt_labels_dir, track_labels_dir):
    gt_labels = load_labels(gt_labels_dir)
    track_labels = load_labels(track_labels_dir)

    total_rmse = 0.0
    total_mae = 0.0
    total_iou = 0.0
    total_accuracy = 0.0
    num_frames = len(gt_labels)
    count = 0

    for frame_id in range(1, num_frames + 1):
        gt_objects = gt_labels.get(frame_id, [])
        track_objects = track_labels.get(frame_id, [])

        if gt_objects and track_objects:
            for gt_object, track_object in zip(gt_objects, track_objects):
                count += 1
                if len(track_object) == 6 and len(gt_object) == 5:
                    gt_x, gt_y, gt_w, gt_h = gt_object[1:]
                    track_x, track_y, track_w, track_h = track_object[1:5]

                    # Calculating RMSE
                    rmse = np.sqrt((gt_x - track_x) ** 2 + (gt_y - track_y) ** 2)
                    total_rmse += rmse

                    # Calculating MAE
                    mae = np.abs(gt_x - track_x) + np.abs(gt_y - track_y)
                    total_mae += mae

                    # Calculating IOU
                    x1, y1 = max(gt_x, track_x), max(gt_y, track_y)
                    x2, y2 = min(gt_x + gt_w, track_x + track_w), min(gt_y + gt_h, track_y + track_h)
                    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                    gt_area = gt_w * gt_h
                    track_area = track_w * track_h
                    iou = inter_area / (gt_area + track_area - inter_area)
                    total_iou += iou

                    # Calculating accuracy
                    if iou >= 0.5:  # Threshold for considering a detection as correct
                        total_accuracy += 1

    avg_rmse = total_rmse / count if count > 0 else 0.0
    avg_mae = total_mae / count if count > 0 else 0.0
    avg_iou = total_iou / count if count > 0 else 0.0
    accuracy = total_accuracy / count if count > 0 else 0.0

    print(f"Average RMSE: {avg_rmse}")
    print(f"Average MAE: {avg_mae}")
    print(f"Average IOU: {avg_iou}")
    print(f"Accuracy: {accuracy * 100}%")

    return avg_rmse, avg_mae, avg_iou, accuracy


def main():
    gt_labels_dir = "dataset/10_txt"
    track_labels_dir = "runs/track/exp9/labels"

    average_rmse, average_mae, average_iou, accuracy = calculate(gt_labels_dir, track_labels_dir)

if __name__ == "__main__":
    main()
