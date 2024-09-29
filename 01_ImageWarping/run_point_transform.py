import cv2
import numpy as np
import gradio as gr
import torch

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

# def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
#     """ 
#     Return
#     ------
#         A deformed image.
#     """
#     global points_src,points_dst
#     n = len(points_src)
#     tensor_src = torch.tensor(points_src)
#     tensor_dst = torch.tensor(points_dst)
#     map1 = torch.zeros(image.shape[0],image.shape[1]) # 
#     map2 = torch.zeros(image.shape[0],image.shape[1])
#     p_head = torch.zeros(n,2) 
#     for y in range(image.shape[0]):
#         for x in range(image.shape[1]):
#             w = torch.zeros(n)
#             p_star = torch.zeros(1,2)
#             q_star = torch.zeros(1,2)
#             p_head = torch.zeros(n,2)
#             q_head = torch.zeros(n,2)
#             A = torch.zeros(2,2)
#             B = torch.zeros(2,2)
#             for i in range (n):
#                 w[i]=1./((tensor_src[i,0]-x)**2+(tensor_src[i][1]-y)**2+eps)**alpha
#                 p_star += w[i]*tensor_src[i,:]
#                 q_star += w[i]*tensor_dst[i,:] 
#             p_star = p_star/torch.sum(w)
#             q_star = q_star/torch.sum(w)
#             for i in range(n):
#                 p_head[i,:]=tensor_src[i,:]-p_star
#                 q_head[i,:]=tensor_dst[i,:]-q_star
#                 A +=torch.matmul(p_head[i,:].unsqueeze(1),p_head[i,:].unsqueeze(0))*w[i]
#                 B +=torch.matmul(p_head[i,:].unsqueeze(1),q_head[i,:].unsqueeze(0))*w[i]
#             A = torch.inverse(A)
#             out_pos = torch.matmul(torch.tensor([[x,y]])-p_star,A)
#             out_pos = torch.matmul(out_pos,B)
#             out_pos = out_pos+q_star
#             map1[y,x]=out_pos[0,0]
#             map2[y,x]=out_pos[0,1]
#     transformed_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
#     warped_image = np.array(transformed_image)
#     ### FILL: 基于MLS or RBF 实现 image warping
#     return warped_image


def image_tranform(image ,map1 , map2):
    image = torch.tensor(image, dtype=torch.float32)
    h,w=image.shape[:2]
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new = torch.tensor(image_new)
    mask = torch.zeros(image_new.shape[:2])
    for y in range(h):
        for x in range(w):
            if 0<=map2[y,x]+pad_size<image_new.shape[1] and 0<=map1[y,x]+pad_size<image_new.shape[0] :
                image_new[int(map2[y,x])+pad_size,int(map1[y,x])+pad_size]=image[y,x]
                mask[int(map2[y,x])+pad_size,int(map1[y,x])+pad_size]=1
    return image_new,mask

# 将for循环改为张量运算，加快运行速度
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Parameters
    ----------
    image: The input image to deform.
    source_pts: The source control points (Nx2 array).
    target_pts: The target control points (Nx2 array).
    alpha: Weight parameter for deformation.
    eps: Small epsilon to avoid division by zero.

    Returns
    -------
    A deformed image.
    """
    for sour,tar in zip(source_pts,target_pts):
        print(sour,tar)
    # Move points to tensors, and move computations to the GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor_dst = torch.tensor(source_pts, device=device, dtype=torch.float32)
    tensor_src = torch.tensor(target_pts, device=device, dtype=torch.float32)
    
    # Ensure h and width are integers
    h, width = image.shape[:2]
    
    # Generate pixel coordinates for the entire image
    xx, yy = torch.meshgrid(torch.arange(width, device=device), torch.arange(h, device=device), indexing='xy')
    pixel_coords = torch.stack([xx, yy], dim=-1).float()  # (h, w, 2)
    #  print(pixel_coords[2,3])
    n = tensor_src.size(0)
    
    # Initialize map1 and map2
    map1 = torch.zeros(h, width, device=device)
    map2 = torch.zeros(h, width, device=device)
    
    # Precompute differences between source points and pixel coordinates
    weights = torch.zeros(h, width, n, device=device)  # Renamed w to weights
    p_star = torch.zeros(h, width, 2, device=device)
    q_star = torch.zeros(h, width, 2, device=device)
    
    # Calculate weight and p_star, q_star
    for i in range(n):
        diff = pixel_coords - tensor_src[i]  # (h, w, 2)
        dist_sq = torch.sum(diff**2, dim=-1) + eps
        weights[..., i] = 1.0 / dist_sq**alpha
        p_star += weights[..., i:i+1] * tensor_src[i]
        q_star += weights[..., i:i+1] * tensor_dst[i]
    
    # Normalize p_star and q_star
    weights_sum = torch.sum(weights, dim=-1, keepdim=True)
    p_star /= weights_sum
    q_star /= weights_sum
    
    # Precompute affine transformations for each pixel
    A = torch.zeros(h, width, 2, 2, device=device)  # (h, w, 2, 2)
    B = torch.zeros(h, width, 2, 2, device=device)  # (h, w, 2, 2)

    for i in range(n):
        # Adjust p_head and q_head to be four-dimensional (h, w, 2)
        p_head = tensor_src[i].unsqueeze(0).unsqueeze(0) - p_star  # Broadcast to (h, w, 2)
        q_head = tensor_dst[i].unsqueeze(0).unsqueeze(0) - q_star  # Broadcast to (h, w, 2)
        
        # Compute A and B, now as four-dimensional tensors
        A += torch.matmul(p_head.unsqueeze(-1), p_head.unsqueeze(-2)) * weights[..., i:i+1, None]
        B += torch.matmul(p_head.unsqueeze(-1), q_head.unsqueeze(-2)) * weights[..., i:i+1, None]
    
    # Inverse of A (h, w, 2, 2)
    A_inv = torch.inverse(A)

    # Compute the affine transformation for each pixel
    transform = torch.matmul(A_inv, B)  # (h, w, 2, 2)
    
    # Apply the deformation
    out_pos = torch.matmul((pixel_coords - p_star).unsqueeze(-2), transform).squeeze(-2) + q_star
    
    map1 = out_pos[..., 0]
    map2 = out_pos[..., 1]
    
    # Move to CPU and convert to numpy for remapping in OpenCV
    map1_cpu = map1.cpu().numpy().astype(np.float32)
    map2_cpu = map2.cpu().numpy().astype(np.float32)
    
    # Use OpenCV to remap the image based on the new coordinates
    transformed_image = cv2.remap(image, map1_cpu, map2_cpu, interpolation=cv2.INTER_LINEAR)
    return transformed_image




def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
