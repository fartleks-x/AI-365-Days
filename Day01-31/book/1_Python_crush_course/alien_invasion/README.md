### 1.源码来源
《Python编程: 从入门到实践》

### 2.打包方法
#### 1. 创建虚拟环境
cd AI-365-Days\Python_crush_course\alien_invasion  
python -m venv venv_pack

#### 2. 激活虚拟环境
venv_pack\Scripts\activate

#### 3. 安装依赖
pip install pygame==2.6.1  
pip install pyinstaller

#### 4. 打包
pyinstaller --onefile --noconsole --add-data "images/*.bmp;images" --collect-all pygame alien_invasion.py

#### 5. 退出虚拟环境
deactivate