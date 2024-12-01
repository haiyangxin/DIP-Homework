import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork, Generator, Discriminator
from torch.optim.lr_scheduler import StepLR
import time

# 设置随机种子,保证每次的训练结果一样
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, 
                   criterion_gan, criterion_l1, device, epoch, num_epochs, lambda_l1=100):
    """
    使用GAN训练一个epoch
    """
    generator.train()
    discriminator.train()
    running_g_loss = 0.0
    running_d_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        batch_size = image_rgb.size(0)
        real_label = torch.ones(batch_size, 1, 15, 15).to(device) * 0.95
        fake_label = torch.zeros(batch_size, 1, 15, 15).to(device)

        # 移动数据到设备
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # 训练判别器
        d_optimizer.zero_grad()
        
        # 真实样本
        real_output = discriminator(image_rgb, image_semantic)
        d_loss_real = criterion_gan(real_output, real_label)
        
        # 生成的假样本
        fake_semantic = generator(image_rgb)
        fake_output = discriminator(image_rgb, fake_semantic.detach())
        d_loss_fake = criterion_gan(fake_output, fake_label)
        
        # 总判别器损失
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        g_optimizer.zero_grad()
        
        # 重新计算判别器对生成图像的输出
        fake_output = discriminator(image_rgb, fake_semantic)
        g_loss_gan = criterion_gan(fake_output, real_label)
        
        # L1损失
        g_loss_l1 = criterion_l1(fake_semantic, image_semantic) * lambda_l1
        
        # 总生成器损失
        g_loss = g_loss_gan + g_loss_l1
        g_loss.backward()
        g_optimizer.step()

        # 更新运行损失
        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

        # 保存样本图像
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_semantic, 'train_results', epoch)

        # 打印训练信息
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], '
                  f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

    return running_g_loss / len(dataloader), running_d_loss / len(dataloader)

def validate(generator, discriminator, dataloader, criterion_gan, criterion_l1, 
            device, epoch, num_epochs, lambda_l1=100):
    """
    验证生成和判别器
    """
    generator.eval()
    discriminator.eval()
    val_g_loss = 0.0
    val_d_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            batch_size = image_rgb.size(0)
            real_label = torch.ones(batch_size, 1, 15, 15).to(device)
            fake_label = torch.zeros(batch_size, 1, 15, 15).to(device)

            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # 生成图像
            fake_semantic = generator(image_rgb)
            
            # 计算判别器损失
            real_output = discriminator(image_rgb, image_semantic)
            fake_output = discriminator(image_rgb, fake_semantic)
            
            d_loss_real = criterion_gan(real_output, real_label)
            d_loss_fake = criterion_gan(fake_output, fake_label)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            # 计算生成器损失
            g_loss_gan = criterion_gan(fake_output, real_label)
            g_loss_l1 = criterion_l1(fake_semantic, image_semantic) * lambda_l1
            g_loss = g_loss_gan + g_loss_l1

            val_g_loss += g_loss.item()
            val_d_loss += d_loss.item()

            # 保存验证集图像
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, fake_semantic, 'val_results', epoch)

    avg_g_loss = val_g_loss / len(dataloader)
    avg_d_loss = val_d_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation, D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}')

def main():
    """
    主函数设置训练和验证过程
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 始化数据集和数据加载器
    train_dataset = FacadesDataset(list_file='/root/autodl-tmp/DIP_Homework/02_DIPwithPyTorch/Pix2Pix/train_list.txt')
    val_dataset = FacadesDataset(list_file='/root/autodl-tmp/DIP_Homework/02_DIPwithPyTorch/Pix2Pix/val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 损失函数
    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    # 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 学习率调度器
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer,
        T_max=200,
        eta_min=1e-6
    )
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        d_optimizer,
        T_max=200,
        eta_min=1e-6
    )

    # 训练循环
    num_epochs = 200
    for epoch in range(num_epochs):
        train_g_loss, train_d_loss = train_one_epoch(
            generator, discriminator, train_loader, 
            g_optimizer, d_optimizer, criterion_gan, criterion_l1, 
            device, epoch, num_epochs, lambda_l1=50
        )
        
        validate(
            generator, discriminator, val_loader, 
            criterion_gan, criterion_l1, device, epoch, num_epochs
        )

        # 更新学习率 - 使用训练损失作为指标
        g_scheduler.step()  # 余弦退火不需要指标
        d_scheduler.step(train_d_loss)  # 传入判别器的训练损失作为指标

        # 保存模型
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'epoch': epoch
            }, f'checkpoints/pix2pix_gan_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
