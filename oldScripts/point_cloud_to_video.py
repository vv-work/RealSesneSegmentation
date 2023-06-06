import open3d as o3d
import numpy as np
import os
import time
import cv2

path_pcd = './pointclouds3/'
list_pcd = os.listdir(path_pcd)

# sort list files
list_pcd = sorted(list_pcd, key=lambda x: int(x.split('frame')[1].split('.pcd')[0]))
print(list_pcd)
vis = o3d.visualization.Visualizer()
vis.create_window()

# define camera options



# define rendering options
render = vis.get_render_option()
render.background_color = [0,0,0]
render.point_size = 1

for i, pcd_name in enumerate(list_pcd):
    pcd = o3d.io.read_point_cloud(path_pcd + pcd_name)
    vis.clear_geometries()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f'./videos/point_cloud_render3/frame{i}.jpg')

vis.destroy_window()

# save video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
size_frame = cv2.imread('./videos/point_cloud_render3/frame0.jpg').shape
out = cv2.VideoWriter('./videos/kinect_real_time_demo3_pc.mp4', fourcc, 8, (size_frame[1], size_frame[0]))
for i in range(len(list_pcd)):
    out.write(cv2.imread(f'./videos/point_cloud_render3/frame{i}.jpg'))
out.release()
