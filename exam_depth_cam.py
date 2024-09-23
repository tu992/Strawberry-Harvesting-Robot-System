# Type help("robolink") or help("robodk") for more information
# Press F5 to run the script
# Documentation: https://robodk.com/doc/en/RoboDK-API.html
# Reference:     https://robodk.com/doc/en/PythonAPI/index.html
#
# This example shows how to retrieve and display the 32-bit depth map of a simulated camera.

from robodk.robolink import *  # RoboDK API

from tempfile import TemporaryDirectory
import numpy as np
from matplotlib import pyplot as plt
import sys
import io

# Đặt mã hóa mặc định thành UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#----------------------------------
# Get the simulated camera from RoboDK
RDK = Robolink()

cam_item = RDK.Item('Depth Camera', ITEM_TYPE_CAMERA)
if not cam_item.Valid():
    cam_item = RDK.Cam2D_Add(RDK.ActiveStation(), 'DEPTH')
    cam_item.setName('Depth Camera')
cam_item.setParam('Open', 1)

#----------------------------------------------
# Get the image from RoboDK
td = TemporaryDirectory(prefix='robodk_')
tf = td.name + '/temp.grey32'
if RDK.Cam2D_Snapshot(tf, cam_item) == 1:
    grey32 = np.fromfile(tf, dtype='>u4')
    w, h = grey32[:2]
    grey32 = np.flipud(np.reshape(grey32[2:], (h, w)))
else:
    raise

# Chụp bản đồ độ sâu từ camera
if not cam_item.Valid():
    print('Không tìm thấy camera')
else:
    print('Đã tìm thấy camera:', cam_item.Name())

    # Chụp ảnh và bản đồ độ sâu từ camera ảo
    snapshot_file = RDK.Cam2D_Snapshot('', cam_item)  # Chụp ảnh từ camera ảo

    if snapshot_file:
        # print('Đã lưu ảnh chụp tại:', snapshot_file)
        
        # Đọc file độ sâu (nếu file chứa dữ liệu độ sâu)
        depth_map = np.loadtxt(snapshot_file, delimiter=',')  # Giả định dữ liệu độ sâu được lưu trong file .csv

        # Kiểm tra kích thước của bản đồ độ sâu
        height, width = depth_map.shape
        print(f'Kích thước bản đồ độ sâu: {width}x{height}')
        
        # Chọn một điểm ảnh cụ thể để lấy giá trị độ sâu (ví dụ: điểm (100, 200))
        x, y = 100, 200

        if 0 <= x < width and 0 <= y < height:
            depth_value = depth_map[y, x]  # Lấy giá trị độ sâu từ điểm (x, y)
            print(f'Giá trị độ sâu tại điểm ảnh ({x}, {y}): {depth_value}')
        else:
            print(f'Điểm ảnh ({x}, {y}) nằm ngoài giới hạn của bản đồ độ sâu ({width}x{height}).')
    else:
        print('Không thể chụp ảnh từ camera.')
#----------------------------------------------
# Display
grey32[grey32 == 0] = 2**32 - 1  # This is for display purposes only! Values of 0 do not have any measurements.
plt.imshow(grey32, 'gray')
plt.show()