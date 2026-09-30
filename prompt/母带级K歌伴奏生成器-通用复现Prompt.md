# 提示词：母带级 K 歌伴奏生成器（通用复现 Prompt）

> 用途：把下面整段复制给任意一个能写代码的 AI（GPT / Claude / 豆包 / DeepSeek 等），
> 要求它生成一个功能相同、可直接本地运行的 Gradio Web App。本提示词已把踩过的坑、
> API 细节、质量要点全部写清，目标是让 AI 
>
> **一次到位**
>
> ，无需反复试错。



***

## 你是一名资深的 Python 音频处理工程师

请为我编写并交付一个基于 **Gradio** 的本地 Web App，名叫「母带级 K 歌伴奏生成器」。

目标：用户上传一首歌的原唱，App 自动把**所有人声（含主唱、和声、第二音部）彻底去掉**，

对**纯乐器部分**做半音变调，最后输出一首干净、不爆音、音质尽量不损失的变调伴奏。

全程本地运行，不开服务器、不调远程 API。



***

## 一、技术栈（依赖清单）

请在代码开头注释里写清楚安装命令：



```
pip install gradio demucs audio-separator pyrubberband soundfile numpy
```

额外要点：



* 用 `subprocess` 调用 demucs 的 CLI（`python -m demucs`），不要用它的 Python API（API 易变）。

* 音轨分离模型用 **Demucs**；二次去人声用 **audio-separator**（内部是 MDX-Net）。

* 变调用 **pyrubberband**（依赖本地 rubberband CLI）。

* 音频读写用 **soundfile**，数值运算用 **numpy**。



***

## 二、核心业务流水线（必须严格按这个顺序）



1. 用户上传原唱，选择**变调半音数**（范围 -12 \~ +12）。

2. **第一步 粗分**：用 Demucs 把原唱分离为多个分轨。

* 默认用 **6 分轨模型&#x20;**`htdemucs_6s`：输出 `vocals / drums / bass / guitar / piano / other`。

* 另提供 4 分轨选项：`htdemucs`（标准）与 `htdemucs_ft`（fine-tuned 高保真）。

* **注意：demucs 官方不存在&#x20;**`htdemucs_ft_6s`**，也没有比 6 更细的官方模型**，不要写进选项。

1. **第二步 丢弃与清洗**：

* **直接丢弃&#x20;**`vocals`**&#x20;轨**（主唱和和声都在里面）。

* `other` 轨**可选**用 audio-separator 做一次「去人声（No Vocals）」二次清洗，兜住可能漏进乐器轨的和声。

* 提供一个开关让用户选择是否二次清洗：**开启更彻底去和声，但乐器（尤其与主唱同频段的泛音）可能略变薄；关闭最大保真**。

1. **第三步 分轨变调**：

* 对除 vocals 外的所有乐器轨（6 分轨就是 drums /bass/guitar /piano/other）做**相同半音数**的变调。

* 提供两种变调方式，**默认 “合并后一次性变调”**（更高保真，见下）：


  * **合并后一次性变调**：先把所有乐器轨用 numpy 按列相加合成一条「纯伴奏」，再**整体变调一次**。相位更一致、多次拉伸叠加的伪影更少。

  * **分轨独立变调**：每轨各自变调后再合并。保留分轨独立性，但多次独立拉伸会叠加相位伪影。

1. **第四步 合并 + 归一化 + 响度增强**：

* 用 numpy 按列相加（必须处理 shape 不一致：**长度对齐到最长并补零；单声道统一转双声道**）。

* **峰值归一化**到 0.9 左右，防止爆音。

* **可选响度增强**（dB），但**绝不能削波 / 出爆破音**：实现一个 “软膝峰值限制器”，超过阈值的部分平滑压缩到上限，永远不超过上限（见代码示例）。



***

## 三、UI 需求（Gradio）



* 左侧控件：


  * 上传音频（`wav / mp3 / flac / m4a` 等，`type="filepath"`）。

  * 变调半音数 Slider（-12 \~ +12）。

  * 音量增强 Slider（-6 \~ +24 dB，0 = 不变）。

  * 分离模式 Radio：`标准4分轨(htdemucs)` / `高保真4分轨(htdemucs_ft)` / `六分轨(htdemucs_6s)`（默认）。

  * 二次人声清洗 Checkbox（默认**关闭**，配无损源保真最好）。

  * 变调方式 Radio：`合并后一次性变调`（默认） / `分轨独立变调`。

  * 「生成伴奏」按钮。

* 右侧：实时**状态 Markdown**（分步提示：粗分→清洗→变调→合并→写出）+ 结果 Audio 播放器。

* 因为要跑两次 AI 分离 + 变调，耗时较长，**必须做好详细进度提示**（用 `gr.Info` / `gr.Progress` / 更新 Markdown），不要让人干等。

* 首次运行会下载模型，要提示 “耗时较长属正常”。



***

## 四、关键实现细节与踩坑经验（务必遵守）

### 1. Demucs 模型名与分轨收集



* 分轨通过 `subprocess.run(["demucs", "-n", model, "--out", out_dir, input])` 调用。

* 输出目录结构：`<out_dir>/<model>/<输入文件名>/<stem>.wav`。

* **不要硬编码分轨名单**：分离后**读取该目录下所有&#x20;**`*.wav`**，按文件名（**`stem`**）收集成 dict**，再丢掉 `vocals`。这样 4 轨和 6 轨都通用。

* 若目录不存在或没有 wav，要抛出带实际目录内容的报错。

### 2. audio-separator 的 API（0.47.x 大改）



* **构造&#x20;**`Separator`**&#x20;时不传模型名和设备**（设备自动检测 CUDA/CPU）：



```
separator = Separator(output_dir=str(work_dir), output_format="FLAC")
separator.load_model("UVR_MDXNET_KARA_2.onnx")
created = separator.separate(str(other_path))
```



* `load_model`**&#x20;必须传 “模型文件名”**（`UVR_MDXNET_KARA_2.onnx`），传友好名（如 `UVR-MDX-NET Karaoke 2`）会报 not found。

* `separate()`**&#x20;返回的是裸文件名（相对当前工作目录）**，实际文件写在 `output_dir` 里，要自己拼绝对路径再判断存在性。

* 从返回结果里挑出 “伴奏 / Instrumental” 轨（文件名含 `instrumental` / `no_vocal(s)`），而不是 `vocals` 轨。

* 中间轨用 **FLAC** 无损输出（不要 WAV 16bit，会损失）。

### 3. pyrubberband /rubberband 变调



* 直接调用 `rb.pitch_shift(data, sr, n_steps=semitones, rbargs={...})`。

* **强制使用 R3 精细引擎**：rubberband 以 `rubberband` 名调用时默认是 **R2 快速引擎（**`-2 --fast`**，伪影多）**，必须传 `rbargs={"-3": ""}` 切到 **R3 精细引擎（**`-3 --fine`**）**，音质显著更好，代价是更慢。

* **Windows 下&#x20;**`pip install rubberband`**&#x20;源码会编译失败**（缺 librubberband）。正确做法：下载官方 Windows 可执行包

  `https://breakfastquay.com/files/releases/rubberband-<版本>-gpl-executable-windows.zip`，

  解压出 `rubberband.exe`（需同目录 `sndfile.dll`），并在程序启动时把该目录注入 PATH，实现开箱即用。

* `pitch_shift` 的 `rbargs` 键 / 值会原样拼进 CLI，单值开关传空字符串值。

### 4. 音频格式与数值



* 全程用 **float32** 处理。

* 中间 other 清洗输出 **FLAC**（无损）。

* **最终输出写 32-bit float（**`subtype="FLOAT"`**）**，避免被量化成 16-bit 损失动态。

* 采样率统一用 44100Hz（demucs 内部固定 44.1k）。

* 合并时：`align_and_sum` 统一转双声道、长度补零到最长、按列求和。

### 5. 归一化 / 响度增强 / 防爆音

峰值归一化 + 软膝限制器示例：



```
def peak_normalize(data, target_peak=0.9):
    peak = float(np.max(np.abs(data)))
    if peak <= 1e-9:
        return data
    return (data * (target_peak / peak)).astype(np.float32)

def soft_limiter(data, ceiling=0.95, knee=0.25):
    T = max(ceiling - knee, 0.0)
    amp = np.abs(data)
    limited = np.where(amp <= T, amp, ceiling - knee * np.exp(-(amp - T) / knee))
    return np.sign(data) * limited.astype(np.float32)

def apply_loudness_boost(data, boost_db, ceiling=0.95):
    if boost_db <= 0:
        return data
    gain = 10 ** (boost_db / 20.0)
    return soft_limiter(data * gain, ceiling=ceiling)
```

要点：`boost_db <= 0`**&#x20;原样返回；提高音量只靠软膝压缩，永不削波、不出爆破音**。



***

## 五、工程质量要求（工程化）



1. **临时文件管理**：demucs 和 audio-separator 会产生大量中间 wav/flac。必须用

   `tempfile.TemporaryDirectory()` 承载全部中间产物，处理结束**自动彻底清理**，只把最终结果写到持久输出目录。

2. **异常与对齐**：合并前检查 numpy shape；长度不一截断 / 补零，声道数不一一律转双声道。

3. **模块化单文件**：输出一个 `app.py`，拆成 “基础工具 / Demucs 分离 / MDX 清洗 / 变调 / 主流程 / Gradio UI” 几段，加详细中文注释。

4. **状态提示**：分步回传进度文本，界面实时更新。

5. **启动自检**：程序启动时自动把 rubberband 目录加入 PATH，并校验关键依赖存在，缺失时报清晰的安装指引。



***

## 六、验收标准

交付后，我能做到：



* 一行命令 `python app.py` 启动，浏览器打开本地 URL 即出现界面。

* 上传一首歌（最好无损 FLAC 源），默认设置（六分轨 + 关二次清洗 + 合并后一次性变调）能产出：


  * 无人声残留（或仅极轻微和声）的伴奏；

  * 变调正确、音准不飘；

  * 无爆音 / 破音 / 削波；

  * 输出是 32-bit float / 44100Hz / 立体声；

  * 大文件也流畅（中间文件自动清理，不堆积）。

## 七、交付物



* 单文件 `app.py`（可直接运行）。

* 说明如何装依赖、如何拿到 / 放置 rubberband.exe、首次运行会下载哪些模型。



***

### 附：给复现 AI 的一句话提示

> 请严格按上面的流水线顺序实现，重点落实三件事：① 6 分轨分离并丢弃 vocals；② 变调强制 R3 精细引擎；③ 输出 32-bit float 且合并时做好对齐与防爆音。不要简化跳过任何一步。