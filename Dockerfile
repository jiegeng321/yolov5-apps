FROM registry.tongdun.me/xdsec/cv.tensorrt.base:2.1

ARG APPNAME=ai-brand-logo
ENV APPNAME=${APPNAME} \
    PYTHONIOENCODING="UTF-8" \
    TZ=Asia/Shanghai

# RUN useradd admin
# USER admin

# 工作目录/home/admin/应用目录，不要改
WORKDIR /home/admin/$APPNAME

# 放入整个工程
ADD . .

# 运行启动脚本
CMD ["bash", "start.sh"]

EXPOSE 8088

