import os
import cv2
import numpy as np
from server.vision.TestVedio_v3 import get_key_points_v3_safe
from server.utils.io_utils import read_cmd


def visualize_skeleton(video_path, output_path, kp_list):
    # 打开视频源
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频源: " + str(video_path))

    # 获取视频属性（用于保存输出）
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频写入器（如果需要保存）
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编解码器（根据系统调整）
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    it = 0

    # 处理每一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束或读取失败

        # ====== 图形绘制 ======
        # 绘制圆形（参数检查示例）
        for i in range(8):
            # print((kp_list[it][i][0], kp_list[it][i][1]))
            cv2.circle(
                img=frame,
                center=(int(kp_list[it][i][0]), int(kp_list[it][i][1])),
                radius=3,
                color=(0, 0, 255),
                thickness=-1
            )

        # 绘制直线（自动处理越界坐标）
        vec_list = [[4, 7], [7, 6], [6, 5]]
        for i in range(3):
            stard_id = vec_list[i][0]
            end_id = vec_list[i][1]
            start_pt = (int(kp_list[it][stard_id][0]), int(kp_list[it][stard_id][1]))
            end_pt = (int(kp_list[it][end_id][0]), int(kp_list[it][end_id][1]))
            cv2.line(
                img=frame,
                pt1=start_pt,
                pt2=end_pt,
                color=(0, 255, 0),
                thickness=3
            )

        # ====== 显示/保存帧 ======
        # cv2.imshow('Processed Video', frame)
        if writer:
            writer.write(frame)
        it += 1
        # 退出条件（按Q键退出）
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 释放资源
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def kalman_smooth_2d(obs, dt=1.0, q_pos=1e-3, q_vel=1e-4, r_meas=0.5):
    """
    obs: (T, 2) array, 每行是 [x, y]
    返回: smooth: (T, 2)
    """
    T = obs.shape[0]
    # 状态维度 4: [x, y, vx, vy]
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)

    Q = np.diag([q_pos, q_pos, q_vel, q_vel])
    R = np.diag([r_meas, r_meas])

    # 初始化
    x_filt = np.zeros((T, 4))
    P_filt = np.zeros((T, 4, 4))

    x_pred = np.zeros(4)
    x_pred[0:2] = obs[0]      # 初始位置
    P_pred = np.eye(4)

    # 前向滤波
    for t in range(T):
        # 观测更新
        z = obs[t]
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ (z - H @ x_pred)
        P_upd = (np.eye(4) - K @ H) @ P_pred

        x_filt[t] = x_upd
        P_filt[t] = P_upd

        # 时间更新（跳到 t+1 的预测）
        x_pred = A @ x_upd
        P_pred = A @ P_upd @ A.T + Q

    # RTS 向后平滑
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in reversed(range(T-1)):
        P_f = P_filt[t]
        P_fp1 = A @ P_f @ A.T + Q
        C = P_f @ A.T @ np.linalg.inv(P_fp1)
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t+1] - A @ x_filt[t])
        P_smooth[t] = P_f + C @ (P_smooth[t+1] - P_fp1) @ C.T

    return x_smooth[:, :2]   # 只要平滑后的位置

def smooth_all_keypoints(traj):
    # traj: (T, 8, 2)
    T, K, _ = traj.shape
    traj_smooth = np.zeros_like(traj, dtype=float)
    for k in range(K):
        traj_smooth[:, k, :] = kalman_smooth_2d(traj[:, k, :])
    return traj_smooth


def adj_matrix():
    kp_dir = r"D:\04_code\Superowl\server\dataset\data_generator\key_points"
    rat_adj = [
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,0],
        [0,0,1,1,0,1,0,1],
        [1,1,0,0,1,0,1,0],
    ]
    for lst in rat_adj:
        lst = np.array(lst)
    rat_adj = np.array(rat_adj)
    np.save(os.path.join(kp_dir, "adj.npz"), rat_adj)


if __name__ == "__main__":
    # adj_matrix()
    # length_list = [7293, 4312, 5380, 3653, 3089]
    video_dir = r"D:\04_code\Superowl\server\dataset\data_generator\ver2\videos"
    kp_dir = r"D:\04_code\Superowl\server\dataset\data_generator\ver2\key_points\single_kp"
    output_dir = r"D:\04_code\Superowl\server\dataset\data_generator\ver2\labeled_videos"
    cmd_dir = r"D:\04_code\Superowl\server\dataset\data_generator\ver2\cmd"

    full_kp_path = os.path.join(kp_dir, "flow_11030.npy")
    full_cmd_path = os.path.join(kp_dir, "cmd_11030.npy")
    valid_data_range = [(270, 4509),   (4779, 8520),  (8790, 11351), (11621, 14709),(14979, 16797),
                        (17067, 19899),(20169, 23437),(23707, 28388),(28658, 31119),(31389, 33793),
                        (34063, 38505),(38775, 42586),(42856+90, 46223),(46493, 48408),(48678, 51373),
                        (51643, 53895),(54165, 57262),(57532, 60665),(60935, 63529),(63799, 65144),
                        (65414, 68826),(69096, 73127),(73397+270, 75690),(75960+270, 77964-144),(78234+270, 80402-54),
                        (80672+270, 83027),(83297+270, 84907),(85177+270, 87303),(87573, 89142),(89412, 90217),
                        (90487, 91971),(92241+270, 96903),(97173+270, 100611-126),(100881+360, 106107-90),(106377, 110168-234),
                        (110438+270, 112807-162)]

    kp_lists = []
    cmd_lists = []
    tot_length = 0
    print("[", end="")
    for file in os.listdir(video_dir):
        print(file)
        video_path = os.path.join(video_dir, file)
        output_path = os.path.join(output_dir, file[:-4]+".mp4")

        kp_path = os.path.join(kp_dir, "{}.npy".format(file[:-4]))
        cmd_path = os.path.join(cmd_dir, "{}.txt".format(file[:-4]))

        kp_list = get_key_points_v3_safe(video_path=video_path, key_points_path=kp_path)
        kp_list = kp_list.transpose(0, 2, 1)

        # 增加大鼠路径的滤波平滑
        kp_list = smooth_all_keypoints(kp_list)
        # 前30s大鼠还没进入画面，或还没开始移动，在数据中保留，但使用时裁掉
        # kp_list = kp_list[540:]

        # print("kp shape = {}".format(kp_list.shape))
        origin_cmd_list = read_cmd(cmd_path)
        cmd_list = []
        action_map = {"None": 0, "go_ahead": 1, "turn_left": 2, "turn_right": 3}
        for item in origin_cmd_list:
            action, voltage = 0, 0
            if item["tar_action"] != 'None':
                action = action_map[item["final_action"]]
                voltage = item["voltage"]
                if voltage == "None":
                    voltage = 0
            cmd_list.append(np.array([action, voltage]))
        visualize_skeleton(video_path, output_path, kp_list)
        # print("file: {} kp length: {}".format(file, len(kp_list)))
        # print("({}, {}),".format(tot_length+270, tot_length + len(kp_list)), end="")
        tot_length += len(kp_list)
        kp_lists.extend(kp_list)
        cmd_lists.extend(cmd_list)
    print("]")
    print("kp length: {}".format(len(kp_lists)))
    np.save(full_kp_path, kp_lists)
    np.save(full_cmd_path, cmd_lists)
    # break
