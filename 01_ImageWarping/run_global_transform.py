import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])


def image_interpolation(image):
    # 获取原图像的高度、宽度和通道数
    height, width, channels = image.shape
    
    # 创建一个与原图像同样大小的输出图像，用于存储结果
    output_image = np.zeros_like(image)
    
    # 遍历每个像素
    for y in range(height):
        for x in range(width):
            # 初始化周围像素值的累加器
            pixel_sum = np.zeros(channels)
            count = 0
            
            # 遍历3x3的邻域
            for dy in range(-1, 2):  # -1, 0, 1
                for dx in range(-1, 2):  # -1, 0, 1
                    ny = y + dy
                    nx = x + dx
                    
                    # 检查邻域坐标是否在图像范围内
                    if 0 <= ny < height and 0 <= nx < width:
                        pixel_sum += image[ny, nx]
                        count += 1
            
            # 计算平均值
            output_image[y, x] = pixel_sum / count
    
# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    # image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    # image = np.array(image_new)
    # transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    
    # Get the center of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

    # Create the transformation matrix
    transformation_matrix = np.eye(3)

    # Translate to center
    translation_to_center = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    
    # Scaling
    scaling_matrix = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])

    # Rotation
    theta = np.radians(rotation)  # Convert degrees to radians
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Combine transformations: first translate to center, then scale, then rotate, and finally translate back
    transformation_matrix = np.dot(translation_to_center, transformation_matrix)
    transformation_matrix = np.dot(scaling_matrix, transformation_matrix)
    transformation_matrix = np.dot(rotation_matrix, transformation_matrix)

    # Translate back to original position
    translation_back = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    transformation_matrix = np.dot(translation_back, transformation_matrix)

    # Apply additional translation
    additional_translation = np.array([
        [1, 0, translation_x+pad_size],
        [0, 1, -translation_y+pad_size],
        [0, 0, 1]
    ])
    transformation_matrix = np.dot(additional_translation, transformation_matrix)
    transformation_matrix_inver = np.linalg.inv(transformation_matrix)

    # Perform the transformation
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         # Convert to homogeneous coordinates
    #         coords = np.array([x, y, 1])
    #         new_coords = np.dot(transformation_matrix, coords) # 矩阵乘法

    #         # Get new coordinates
    #         new_x, new_y = int(new_coords[0]), int(new_coords[1])

    #         # Check bounds and assign pixel value
    #         if 0 <= new_x+pad_size < image_new.shape[1] and 0 <= new_y+pad_size < image_new.shape[0]:
    #             image_new[new_y+pad_size, new_x+pad_size] = image[y, x]
    output_image=cv2.warpAffine(image,transformation_matrix[:2,:],(image_new.shape[1],image_new.shape[0]),flags=cv2.INTER_LINEAR)
    # 考虑反向映射
    # for y in range(image_new.shape[0]):
    #     for x in range(image_new.shape[1]):
    #         coords = np.array([x-pad_size,y-pad_size,1])
    #         new_coords = np.dot(transformation_matrix_inver,coords)
    #         new_x,new_y = int(new_coords[0]),int(new_coords[1])
    #         if 0<=new_x<image.shape[1] and 0<=new_y<image.shape[0]:
    #             image_new[y,x] = image[new_y,new_x] 

    # Flip the image horizontally if needed
    if flip_horizontal:
        # image_new = np.flip(image_new, axis=1) #水平翻转
        output_image = np.flip(output_image,axis=1)
    # transformed_image=np.array(image_new)
    return np.array(output_image)

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
