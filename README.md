# POD 印花提取

从实拍照片中提取印花图案，输出带真实 alpha 通道的 RGBA PNG，用于跨境 POD 印刷。

本项目的核心目标是**印刷级保真**：不做生成式重绘，不使用任何图片大模型，全部由经典图像处理算法完成。因此输出结果可复现、可解释，文字、logo、半调网点、签名等细节保留源图内容。

## 为什么不用图片生成模型

gpt-image-2 一类的生成式模型输出的是 RGB 图像，没有真正的透明通道，边缘只是概率采样结果。白色衬衫背景会让边缘混入白色，表现为白点、白边；同时模型会"合理重画"小字、赞助商 logo 和网点，导致内容与源图不一致，印到衣服上即为废品。

消除白边的关键在于**把 alpha 与观测颜色解耦，并反解前景色**，这是数学问题而非模型能力问题。因此引入模型只会增加不可控环节，没有对应收益。

## 环境要求

- Python 3.11.9
- Windows / Linux / macOS 均可，依赖为纯 wheel 包

依赖库见 `requirements.txt`：

```
numpy==2.2.6
scipy==1.13.1
opencv-python-headless==5.0.0.93
```

使用 headless 版 OpenCV，不含 GUI 依赖，`cv2.imshow` 等窗口函数不可用。当前脚本只做读写与计算，不受影响。

## 安装

```powershell
# 创建虚拟环境（Windows 下指定 3.11.9 解释器）
D:\develop\python3119\python.exe -m venv .venv

# 安装依赖
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 快速开始

测试样本放在 `test-data/` 目录下，该目录不入库，需要自行放入 `002.jpeg` 与 `002_expected.png`。输出默认写入 `output/` 目录，同样不入库。

```powershell
.\.venv\Scripts\python.exe extract_print.py test-data/002.jpeg
```

输出示例：

```
[1/8] input 414x460
[2/8] background estimated, trusted bg ratio 0.379
[3/8] score percentiles 0.468 0.877 0.934 0.987 1.000 1.000
[4/8] coarse mask ratio 0.632
[5/8] holes filled ratio 0.0394
[6/8] alpha solved
[7/8] alpha semi-transparent ratio 0.0095
[8/8] saved output/002_out.png 901x1024, 1.2s
```

需要与参考图 `test-data/002_expected.png` 同尺寸时，指定长边：

```powershell
.\.venv\Scripts\python.exe extract_print.py test-data/002.jpeg --long-side 1557
```

## 处理流程

脚本按 8 个阶段串行执行，每个阶段对应一个函数。

### 1. 读取图像

`read_image` 用 `np.fromfile` + `cv2.imdecode` 读取，规避 Windows 下中文路径问题，内部统一转为 BGR float32（0~1 区间）。

### 2. 背景建模

`estimate_background` 解决的问题是：衬衫有褶皱，背景亮度并不均匀，不能用固定阈值判断"白色"。

做法是先筛出"够亮且几乎无彩色"的像素作为背景候选，再用 5 阶二维多项式拟合出一张平滑的"空白衬衫底图"。拟合过程用 `robust_fit` 迭代 3 轮，每轮剔除残差偏大的样本，避免误入的图案像素污染背景模型。

### 3. 油墨概率

`ink_score` 将实际颜色与背景模型逐像素相除，得到两个偏离量：

- 亮度偏离：实际比预期暗，说明有墨
- 色度偏离：实际比预期更有颜色，说明有彩墨

两者取最大值得到 0~1 的"油墨浓度图"。注意白墨叠在白布上偏离量为 0，本步骤检测不到，由下一步单独处理。

### 4. 粗蒙版

以 `--thresh-hi` 二值化，闭运算补小缝，`remove_small_regions` 剔除面积小于 `--min-area` 的碎片。

### 5. 封闭孔洞填充

`fill_enclosed` 是处理"白布上的白字"的关键。做法是从图像四条边向内做连通域标记，能连到边界的属于真背景，连不到的是被油墨轮廓包住的孔洞，判定为白墨并填充。面积超过 `--max-hole-ratio` 比例的孔洞不填。

已知副作用：签名圈内的大块留白也会被当作白墨填充，详见 `TODO.md`。

### 6. Alpha 求解

图案内部直接置为 1，外部置为 0，只有边界 `--band-width` 像素宽的窄带需要细算。

窄带内用"最近高置信油墨色"做向量反投影：比较当前像素离纯墨色近还是离布色近，按投影比例得到 alpha。这样 alpha 与观测颜色自洽，数学上不会出现"图案边缘算出半白"的情况。低置信区域（白墨叠白布）回退到粗蒙版结果。求解后再用 `guided_filter` 把边界对齐到真实颜色边缘。

### 7. 颜色替换与清理

这一步是根除白点的核心。边界像素的 RGB 里混着布料白，即使 alpha 正确，放大后仍会露出一圈白边，因此直接把窄带像素颜色整体替换为最近的纯油墨色。同时 `clean_alpha` 只在 alpha 域做杂点与针孔清理，不触碰 RGB，避免把半调网点和小字一起磨掉。

### 8. 裁剪与放大

`crop_to_alpha` 按 alpha 包围盒裁掉空白，`resize_output` 分通道放大：

- 颜色用 Lanczos 保留锐度
- alpha 用 Cubic 更平滑
- 放大后收紧 alpha 的灰度过渡带，减少半透明毛边
- 最后做一次轻度 unsharp mask 提升观感

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `input` | 必填 | 输入实拍图路径 |
| `-o`, `--output` | `output/002_out.png` | 输出 RGBA PNG 路径 |
| `--debug-dir` | `debug` | 中间结果输出目录 |
| `--long-side` | `1024` | 输出长边像素 |
| `--score-sigma` | `1.0` | 油墨概率平滑半径，用于让半调网点连成面 |
| `--thresh-hi` | `0.15` | 粗蒙版二值化阈值，越低保留越多细节、杂点也越多 |
| `--band-width` | `2` | alpha 未知带半径，单位为像素 |
| `--min-area` | `8` | 粗蒙版阶段剔除的最小连通域面积 |
| `--max-hole-ratio` | `0.25` | 封闭孔洞填充的面积上限，占全图比例 |
| `--speck-min` | `12` | alpha 域剔除杂点的最小面积 |
| `--pinhole-max` | `8` | alpha 域填补针孔的最大面积 |
| `--sharpen` | `0.3` | 放大后的锐化强度，0 表示关闭 |
| `--pad` | `8` | 裁剪时四周保留的像素边距 |

`--thresh-lo` 当前版本已声明但未参与计算，为后续分级阈值预留。

## 输出文件

- `output/002_out.png`：最终 RGBA 结果
- `debug/01_background.png`：背景模型，用于确认拟合是否被图案污染
- `debug/02_score.png`：油墨概率图
- `debug/03_mask.png`：粗蒙版（含孔洞填充）
- `debug/04_trimap.png`：三分图，白为确定前景、黑为确定背景、灰为未知带
- `debug/05_alpha.png`：求解得到的 alpha
- `debug/06_check_dark.png`：深色底合成，用于检查白边
- `debug/07_check_magenta.png`：品红底合成，用于检查透明度

调参时应重点看 `06_check_dark.png`，白边在深色底上最明显。

## 验收标准

- 深色底合成下，边缘 1~2 像素环内无可见亮环
- alpha 域无独立的孤立杂点
- 以下 5 个 ROI 在 100% 与 200% 放大下无断裂、无挖空：
  1. 底部白色 1951 / 2001 与灰色徽标边缘
  2. 红色网点底与白色火焰形轮廓交界
  3. 车身黑色包围与浅色织物背景交界
  4. "Dale Earnhardt" 连笔签名最细的笔画
  5. 车身侧面赞助商小字
- 与 `002_expected.png` 并排比对时，只允许存在"源图分辨率不足"的差异，不允许出现内容层面的差异

`002_expected.png` 本身经过重绘或放大，不能作为逐像素金标准，只用于观感与边缘质量参照。

## 已知限制

- **分辨率上限由源图决定**。`002.jpeg` 仅 414×460，放大到 1000 像素以上时锐度低于参考图，这是非生成式放大的物理限制。
- **白墨与衬衫透出存在固有歧义**。单张照片无法区分"图案里的白"和"图案围住的布"，当前策略选择保白墨。若目标面料为白色或浅色，该选择无害。
- **参数按单图调优**。阈值默认值针对 `002.jpeg` 调整，换成别的底色或布料需要重新标定。
- **未做透视矫正**。当前样本基本正对镜头。若源图存在倾斜，需要补一个四角检测与透视变换环节。

## 目录结构

```
.
├── extract_print.py     主流程脚本
├── requirements.txt     依赖清单
├── README.md            项目说明
├── TODO.md              待办事项
├── .gitignore           忽略规则
├── output/              输出结果（已忽略）
│   └── 002_out.png          最终 RGBA 结果
├── debug/               中间结果（已忽略）
├── test-data/           测试样本（已忽略）
│   ├── 002.jpeg             输入样本
│   └── 002_expected.png     参考效果图
└── .venv/               虚拟环境（已忽略）
```

## 待办

见 `TODO.md`。当前有两项：签名圈内留白改回透明、阈值与锐化的观感调优。
