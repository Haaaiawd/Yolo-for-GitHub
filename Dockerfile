# 使用一个官方的 Python 运行时作为父镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 1. 单独复制并安装依赖
# 这一步可以利用 Docker 的层缓存机制，只要 requirements.txt 不变，就不需要重新安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 复制项目的所有其他文件
COPY . .

# 定义默认的入口点。
# 我们将其注释掉，因为推荐通过 `docker run` 命令的参数来执行具体任务，这样更灵活。
# 例如: docker run <image> python train.py --args...
# CMD ["python", "./train_yolov11.py"] 