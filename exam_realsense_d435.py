import pyrealsense2 as rs
import numpy as np

# Bước 1: Khởi tạo pipeline
pipeline = rs.pipeline()

# Bước 2: Cấu hình stream
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Bước 3: Bắt đầu stream
pipeline.start(config)

try:
    while True:
        # Bước 4: Lấy khung hình
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()  # Lấy ảnh độ sâu
        color_frame = frames.get_color_frame()  # Lấy ảnh màu
        
        if not depth_frame or not color_frame:
            continue

        # Bước 5: Chuyển đổi dữ liệu thành mảng NumPy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Bước 6: Lấy khoảng cách tại một điểm cụ thể
        # Lấy thông số nội tại của camera
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Tọa độ pixel muốn chuyển đổi (u, v)
        u, v = 320, 240  # Ví dụ tọa độ giữa khung hình
        
        # Lấy giá trị độ sâu tại tọa độ (u, v)
        Z = depth_frame.get_distance(u, v)
        
        # Chuyển đổi sang tọa độ không gian thực
        X, Y = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], Z)
        
        print(f"Tọa độ thực của điểm (X, Y, Z): ({X}, {Y}, {Z})")
        
finally:
    # Dừng pipeline khi hoàn thành
    pipeline.stop()
