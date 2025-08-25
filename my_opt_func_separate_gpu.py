import torch
import numpy as np
import time
import scipy.io as sio
import mat73
from scipy.linalg import lstsq

def gpu_optimize_pages(J_rate_part, A0, V_part, points_part, lr=0.1, num_iter=200):
    # 转换数据为PyTorch张量并移到GPU
    device = torch.device('cuda')
    J_rate = torch.tensor(J_rate_part, dtype=torch.float32, device=device)
    A0 = torch.tensor(A0, dtype=torch.float32, device=device)
    V = torch.tensor(V_part, dtype=torch.float32, device=device)
    points = torch.tensor(points_part, dtype=torch.float32, device=device)
    
    # 获取维度信息
    num_pages, num_points, _ = J_rate.shape
    
    # 上界矩阵 [batch, 3] (每个页面的A0作为上界)
    upper_bounds = A0[:, 0, :].squeeze(1)  # [batch, 3]
    
    # 随机初始化优化变量 [batch, 3]
    J0_var = (torch.rand(num_pages, 3, device=device) * upper_bounds).clone().detach()
    J0_var.requires_grad = True
    
    # 使用Adam优化器（更稳定）
    optimizer = torch.optim.Adam([J0_var], lr=lr)
    
    for i in range(num_iter):
        # 计算J0_all: [batch, 3] -> [batch, points, 3]
        J0_expanded = J0_var.unsqueeze(1).expand(-1, num_points, -1)
        J0_all = J0_expanded * J_rate
        
        # 计算向量叉积和代价函数
        A0_expanded = A0.expand(-1, num_points, -1)  # [batch, points, 3]
        cross_products = torch.cross(J0_all - A0_expanded, V, dim=-1)
        cost = torch.sum(cross_products ** 2, dim=(1, 2))  # [batch]
        
        # 优化步骤
        optimizer.zero_grad()
        cost.mean().backward()  # 使用平均损失稳定优化
        optimizer.step()
        
        # 应用边界约束 (投影法)
        with torch.no_grad():
            min_val = torch.tensor(0.0, device=upper_bounds.device)  # 确保同设备
            J0_var.data = torch.clamp(J0_var.data, min=min_val, max=upper_bounds)
    
    return J0_var.detach().cpu().numpy()

def main():
    start_time = time.perf_counter()
    # 加载数据
    opt_data_temp = mat73.loadmat('opt_data_temp.mat')
    
    # 获取变量
    J_rate_part = opt_data_temp['J_rate_part']
    A0 = opt_data_temp['A0']
    V_part = opt_data_temp['V_part']
    points_part = opt_data_temp['points_part']

    # import pdb
    # pdb.set_trace()  # 程序运行到此处会暂停并进入交互式调试

    # 调整数据维度为 [batch_size, num_points, 3]
    J_rate_part = np.transpose(J_rate_part, (2, 0, 1))          # [batch, 396, 3]
    A0 = np.expand_dims(A0, axis=[0,1])                            # [1, 1, 3]
    A0 = np.repeat(A0, J_rate_part.shape[1], axis=1)            # [1, 396, 3]
    A0 = np.repeat(A0, J_rate_part.shape[0], axis=0)            # [batch, 396, 3]
    V_part = np.transpose(V_part, (2, 0, 1))                    # [batch, 396, 3]
    points_part = np.transpose(points_part, (2, 0, 1))           # [batch, 396, 3]
    
    # GPU优化
    start_time_gpu = time.perf_counter()
    J0_result = gpu_optimize_pages(J_rate_part, A0, V_part, points_part)
    end_time_gpu = time.perf_counter()
    print(f"GPU优化耗时: {end_time_gpu - start_time_gpu:.2f}秒")
    
    # 保存结果
    J0_result = np.expand_dims(J0_result.T, axis=0)  # [1, 3, batch]
    data_dict = {'J0_result': J0_result}
    sio.savemat('opted_data_temp.mat', data_dict)

    end_time = time.perf_counter()
    print(f"GPU优化耗时: {end_time - start_time:.2f}秒")

if __name__ == '__main__':
    main()