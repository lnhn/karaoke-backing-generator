# 🎤 母带级 K 歌伴奏生成器

一个**完全本地运行**的 Gradio Web App：上传一首歌的原唱，自动去掉**所有人声（含主唱、和声、第二音部）**，对**纯乐器部分**做半音变调，输出干净、不爆音、音质尽量不损失的伴奏。不调任何远程 API，数据不出本机。

---

## ✨ 功能特性

- **AI 音轨分离**：Demucs 分离为 **4 轨或 6 轨**（6 轨默认，单独分出钢琴、吉他、贝斯，去人声时对乐器音色损伤更小）。
- **彻底去人声**：直接丢弃 vocals 轨；`other` 轨**可选**用 MDX-Net（audio-separator）二次清洗，兜住漏网和声。
- **高质量变调**：强制使用 **rubberband R3 精细引擎**（默认 R2 快速引擎伪影多），变调音准不飘。
- **两种变调方式**：默认「合并后一次性变调」（先把乐器合成纯伴奏再整体变调一次，相位更一致、伪影更少）；可选「分轨独立变调」。
- **响度增强防爆音**：音量可 +dB 提升，内置**软膝限制器**，永不削波、不出爆破音。
- **保真输出**：全程 float32，中间轨用无损 FLAC，最终输出 **32-bit float / 44100Hz / 立体声**。
- **工程化**：所有中间文件在临时目录中自动清理，只保留最终结果；分步实时进度提示。

---

## 📦 技术栈

| 组件 | 用途 |
|---|---|
| [Gradio](https://gradio.app/) | Web UI |
| [Demucs](https://github.com/facebookresearch/demucs) | 主分离（`htdemucs` / `htdemucs_ft` / `htdemucs_6s`） |
| [audio-separator](https://github.com/nomoth/cuwave) | other 轨二次去人声（`UVR_MDXNET_KARA_2.onnx`） |
| [pyrubberband](https://github.com/bmcfee/pyrubberband) + rubberband CLI | 变调 |
| soundfile / numpy | 音频读写与数值运算 |

---

## 🔧 环境要求与安装

### 系统要求
- Python **3.10+**（本项目在 3.14 实测通过）
- 建议 8GB+ 内存；CPU 可跑（分离耗时较长），有 NVIDIA GPU + CUDA 会快很多

### 安装依赖

```bash
pip install gradio demucs audio-separator pyrubberband soundfile numpy
```

### 准备 rubberband CLI（Windows 必须手动）

> Windows 下 `pip install rubberband` 源码会编译失败，需要官方 Windows 可执行包。

1. 下载：`https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip`
2. 解压出 `rubberband.exe`（需同目录 `sndfile.dll`），放到项目内如 `rubberband-win/rubberband-4.0.0-gpl-executable-windows/`。
3. 程序启动时会自动把该目录加入 PATH，开箱即用。

### 模型自动下载
首次运行会自动从网上下载 Demucs 与 MDX 模型（`htdemucs_6s` 等），需联网、耗时较长，属正常。

---

## 🚀 启动

```bash
python app.py
```

浏览器打开终端提示的本地地址（默认 `http://127.0.0.1:7860`）。

---

## 🖱️ 使用步骤

1. **上传原唱**：支持 `wav / mp3 / flac / m4a` 等。**推荐无损 FLAC/WAV 源**（有损源会放大分离与变调伪影）。
2. **设置参数**：
   - **变调半音数**：-12 ~ +12（负=降调，正=升调）。
   - **音量增强 dB**：>0 提高响度，自动防爆音。
   - **分离模式**：默认「六分轨(htdemucs_6s)」（推荐）；也可选 4 分轨（更快）。
   - **二次人声清洗**：默认**关闭**（配无损源保真最好）；开启更彻底去和声，但乐器可能略变薄。
   - **变调方式**：默认「合并后一次性变调」（推荐，伪影更少）。
3. 点 **🚀 生成伴奏**，右侧实时显示分步进度。
4. 完成后右侧播放器可直接试听，文件保存在 `outputs/` 目录。

### 推荐配置（配无损源，保真最高）
> 六分轨 + **关**二次清洗 + **合并后一次性变调** + 需要更响则加少量 dB

---

## ⚙️ 工作原理

```
原唱 ──Demucs 分离──▶ vocals / drums / bass / [guitar / piano] / other
                          │（丢弃）
                          ▼
                drums + bass + [guitar + piano] + other(可选MDX清洗)
                          │ 合并成纯伴奏（一次性变调模式）
                          ▼
                pyrubberband 变调（R3 精细引擎）
                          │
                          ▼
                合并 → 峰值归一化 → 软膝响度增强 → 写出 32-bit float
```

---

## ⚠️ 已知限制（如实说明）

- **AI 分离非无损**：Demucs 是神经网络重建，天然有轻微泄漏/薄感，这是本质，参数无法完全消除。
- **demucs 官方 6 分轨是上限**：`htdemucs_6s` 把**电吉他、木吉他合并为一轨 `guitar`**，无法再细分（想更细需 UVR 等付费/手工模型）。不存在 `htdemucs_ft_6s`。
- **源文件是天花板**：原唱本身若是低码率有损（如 128kbps mp3），分离+变调会放大伪影，务必用无损源。
- 二次清洗开启时，与主唱同频段的乐器泛音（钢琴、弦乐）可能被顺带削弱，听感偏“薄”。

---

## 🛠️ 故障排查

| 现象 | 处理 |
|---|---|
| 提示 `htdemucs_ft_6s is neither...` | 该模型不存在，改用 `六分轨(htdemucs_6s)` |
| `load_model ... not found` | audio-separator 需传模型**文件名** `UVR_MDXNET_KARA_2.onnx` |
| 提示找不到 `rubberband` | 确认已放置 `rubberband.exe` + `sndfile.dll`，路径见上文 |
| 处理很慢 | CPU 属正常，6 分轨 + R3 引擎都偏慢；有 GPU 更快 |
| 报错缺依赖 | 按报错提示 `pip install` 对应包 |

---

## 📁 文件结构

```
.
├── app.py                 # 主程序（单文件，模块化+中文注释）
├── README.md              # 本文档
├── outputs/               # 生成结果（32-bit float wav）
├── rubberband-win/        # rubberband CLI（Windows 必需）
├── origial/               # （可选）你的原始音频素材
└── prompt/                # 通用复现 Prompt（可给任意 AI 复刻本 App）
```

---

## ⚖️ 版权声明

请仅对**你自己拥有版权、已获授权，或符合合理使用**的音频使用本工具。请勿用于未经授权的版权内容再分发。
