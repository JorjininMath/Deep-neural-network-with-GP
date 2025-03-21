import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class ToyNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ToyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def main():
    # 检查设备可用性
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    print(f"PyTorch版本: {torch.__version__}")
    print(f"使用设备: {device}")

    # 创建训练数据
    X_train = torch.randn(1000, 10)  # 1000个训练样本，每个样本10个特征
    y_train = torch.randn(1000, 1)   # 1000个训练目标值

    # 创建测试数据
    X_test = torch.randn(200, 10)    # 200个测试样本
    y_test = torch.randn(200, 1)     # 200个测试目标值

    # 将数据移动到指定设备
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # 初始化模型
    model = ToyNeuralNetwork(input_size=10, hidden_size=20, output_size=1)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 100
    print("\n开始训练...")
    train_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_end_time = time.time()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, 本轮用时: {epoch_end_time - epoch_start_time:.2f}秒')

    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    print(f"\n训练完成！总训练时间: {total_train_time:.2f}秒")

    # 测试模型性能
    print("\n开始测试...")
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 在测试时不需要计算梯度
        test_start_time = time.time()
        
        # 进行预测
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        
        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        
        print(f"测试集损失: {test_loss.item():.4f}")
        print(f"预测用时: {test_time:.2f}秒")
        print(f"平均每个样本预测用时: {(test_time/len(X_test))*1000:.2f}毫秒")

if __name__ == "__main__":
    main() 