# -*- coding: utf-8 -*-
"""
母带级 K 歌伴奏生成器 (Karaoke Backing-Track Maker)
====================================================
上传原唱 -> 详细分离音轨 -> 彻底去除所有人声 -> 分轨变调 -> 合并输出纯净伴奏

【依赖安装】
    pip install gradio demucs audio-separator pyrubberband soundfile numpy

    · demucs            : htdemucs 模型，把原唱分离为 4 轨 (vocals, drums, bass, other)
    · audio-separator   : MDX-Net 去人声模型，对 other 轨二次清洗，清除残留和声
    · pyrubberband      : 高质量时间拉伸/变调（音高平移）
    · soundfile/numpy   : 音频读写与数值处理
    · gradio            : 本地 Web 界面

【运行方式】
    python app.py
    然后浏览器打开 http://127.0.0.1:7860

【版权提示】
    仅限对你有合法权利（自有作品 / 已获授权 / 个人合理使用）的音频进行处理，
    请遵守当地著作权法规。本工具不用于规避任何版权保护。

【核心流程】
    1) 用户上传原唱，选择变调半音数 (-12 ~ +12)
    2) 粗分：Demucs 将原唱分离为 vocals / drums / bass / other
    3) 丢弃 vocals（主唱与和声都在里面）
       对 other 轨用 audio-separator 做"去人声(No Vocals)"二次清洗 -> other_clean
    4) 对 drums / bass / other_clean 三轨分别做相同半音数变调 (pyrubberband)
    5) numpy 按列相加合并（自动对齐长度、统一声道数）
    6) 峰值归一化防爆音，输出最终纯净变调伴奏
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # 脚本所在目录
OUTPUT_DIR = BASE_DIR / "outputs"                   # 最终结果输出目录
OUTPUT_DIR.mkdir(exist_ok=True)


def _ensure_rubberband_on_path():
    """
    pyrubberband 依赖外部的 rubberband CLI 可执行文件。
    若项目目录下有自带/解压好的 rubberband.exe（官方 Windows CLI），
    则把其所在目录加入 PATH，保证开箱即用，无需用户手动改系统 PATH。
    找不到时保持原 PATH（此时需用户自行安装 rubberband-cli）。
    """
    for exe in ("rubberband.exe", "rubberband-r3.exe"):
        for cand in (BASE_DIR / "rubberband-win", BASE_DIR):
            hit = cand / exe
            if hit.exists():
                exe_dir = str(hit.parent)
                if exe_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")
                return
    # 兜底：递归找一遍
    for exe in ("rubberband.exe", "rubberband-r3.exe"):
        for hit in BASE_DIR.rglob(exe):
            exe_dir = str(hit.parent)
            if exe_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")
            return


_ensure_rubberband_on_path()

# 计算设备：优先 CUDA，其次 CPU。可在命令行环境变量 DB_DEVICE 覆盖。
DEVICE = os.environ.get("DB_DEVICE", "")
if not DEVICE:
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"

# 人声清洗模型：UVR 的 Karaoke 专用 MDX-Net 模型，对人声/和声/二声部压制极强
# 注意：audio-separator 0.47.0 的 load_model() 按"模型文件名"匹配（UVR_MDXNET_KARA_2.onnx）
SEPARATOR_MODEL = "UVR_MDXNET_KARA_2.onnx"

# 变调半音数范围
SEMITONE_MIN, SEMITONE_MAX = -12, 12

# 合并后峰值目标（归一化防止爆音）
TARGET_PEAK = 0.9

# ---- 音质相关配置 ----
# Demucs 可选模型（均经验证存在，adefossez 命名空间下可解析）。
#   4 分轨：htdemucs / htdemucs_ft        -> vocals, drums, bass, other
#   6 分轨：htdemucs_6s                   -> vocals, drums, bass, guitar, piano, other
# 6 分轨会把钢琴、吉他单独剥出，other 不再吞掉它们，去人声时对乐器音色的损伤更小。
# 注意：demucs 不存在 "htdemucs_ft_6s"，也没有比 6 更细的官方分轨。
DEMUCS_MODELS = {
    "标准 4分轨 (htdemucs)": "htdemucs",
    "高保真 4分轨 (htdemucs_ft)": "htdemucs_ft",
    "六分轨 (htdemucs_6s)": "htdemucs_6s",
}
# 最终输出位深：FLOAT = 32-bit float（无损保留动态，避免 16-bit 量化损失）
OUTPUT_SUBTYPE = "FLOAT"
# 中间 other 轨清洗的输出格式：FLAC 为无损，避免清洗结果被 16-bit 量化
SEPARATOR_OUT_FORMAT = "FLAC"
# 最终输出采样率：demucs 内部固定 44100Hz 处理
OUTPUT_SR = 44100


# ---------------------------------------------------------------------------
# 基础音频工具函数
# ---------------------------------------------------------------------------
def load_audio(path: str) -> tuple[np.ndarray, int]:
    """读取音频，返回 (float32 数据, 采样率)。"""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return data, sr


def to_stereo(data: np.ndarray) -> np.ndarray:
    """
    统一转换为双声道，返回形状 (n_samples, 2) 的 float32 数组。
    - 单声道 (n,)            -> 复制为两列 (n, 2)
    - 立体声 (n, 2)          -> 原样返回
    - 多声道 (n, c)          -> 只保留前两声道
    """
    if data.ndim == 1:
        return np.stack([data, data], axis=1)
    if data.shape[1] == 2:
        return data
    # 超过 2 声道：取前两列
    return data[:, :2].astype(np.float32)


def align_and_sum(*arrays: np.ndarray) -> np.ndarray:
    """
    将多轨音频对齐长度并求和。
    对齐策略：统一转双声道 -> 统一采样率已由上层保证 -> 长度补齐到最长（补零），
    这样不会丢失任何一轨的尾部内容。最后返回 (n, 2) 求和结果。
    """
    # 先统一声道
    stereo = [to_stereo(a) for a in arrays if a.size > 0]

    max_len = max(a.shape[0] for a in stereo)
    out = np.zeros((max_len, 2), dtype=np.float32)
    for a in stereo:
        n = a.shape[0]
        out[:n, :] += a
    return out


def peak_normalize(data: np.ndarray, target_peak: float = TARGET_PEAK) -> np.ndarray:
    """峰值归一化：把全音频最大绝对值缩放到 target_peak，防止爆音。"""
    peak = float(np.max(np.abs(data)))
    if peak <= 1e-9:          # 全零/极静音保护
        return data
    return (data * (target_peak / peak)).astype(np.float32)


def soft_limiter(data: np.ndarray, ceiling: float = 0.95, knee: float = 0.25) -> np.ndarray:
    """
    软膝峰值限制器：把超过阈值 (ceiling - knee) 的部分平滑压缩到 ceiling，
    全程不超过 ceiling，避免硬削波产生的"爆破音/破音"。

    曲线：amp <= T 时透明通过；amp > T 时输出 asymptotically 逼近 ceiling，
    在 T 处连续、可导，听感平滑。
    """
    T = max(ceiling - knee, 0.0)          # 起压阈值
    amp = np.abs(data)
    # 分段平滑压缩：超过阈值后向 ceiling 收敛，永不越界
    limited_amp = np.where(
        amp <= T,
        amp,
        ceiling - knee * np.exp(-(amp - T) / knee),
    )
    return np.sign(data) * limited_amp.astype(np.float32)


def apply_loudness_boost(data: np.ndarray, boost_db: float, ceiling: float = 0.95) -> np.ndarray:
    """
    音量增强（响度提升），防爆破音。
    boost_db <= 0 时原样返回；>0 时先按 dB 做线性增益，再用软膝限制器压到 ceiling，
    确保"提高音量"但绝不削波、不产生爆破音。返回 float32。
    """
    boost_db = float(boost_db)
    if boost_db <= 0.0:
        return data
    gain = 10.0 ** (boost_db / 20.0)      # dB -> 线性倍率
    boosted = data * gain
    return soft_limiter(boosted, ceiling=ceiling)


# ---------------------------------------------------------------------------
# 第一步：Demucs 粗分
# ---------------------------------------------------------------------------
def demucs_separate(input_path: Path, out_dir: Path, model: str = "htdemucs") -> dict[str, Path]:
    """
    调用 demucs 分离音轨（4 分轨或 6 分轨均可）。
    返回 {分轨名: wav 路径}，例如 4 分轨 {'vocals','drums','bass','other'}，
    6 分轨在此基础上多 'guitar'、'piano'。
    """
    # demucs 输出目录结构: <out_dir>/<model>/<输入文件名>/<stem>.wav
    stem_dir = out_dir / model / input_path.stem

    cmd = ["demucs", "-n", model, "--out", str(out_dir)]
    if DEVICE:
        cmd += ["-d", DEVICE]
    cmd.append(str(input_path))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Demucs 分离失败\nSTDOUT: {proc.stdout[-2000:]}\nSTDERR: {proc.stderr[-2000:]}"
        )

    if not stem_dir.exists():
        raise RuntimeError(f"Demucs 未生成分轨目录：{stem_dir}")
    # 读取该模型产出的全部分轨（4 或 6 个，按实际存在收集）
    stems = {p.stem: p for p in stem_dir.glob("*.wav")}
    if not stems:
        raise RuntimeError(f"Demucs 未生成任何分轨：{list(stem_dir.iterdir())}")
    return stems


# ---------------------------------------------------------------------------
# 第二步：用 audio-separator 清洗 other 轨（清除残留和声）
# ---------------------------------------------------------------------------
def clean_other_track(other_path: Path, work_dir: Path) -> Path:
    """
    对 Demucs 的 other 轨做"去人声(No Vocals)"二次清洗。
    返回清洗后的纯乐器轨 other_clean.wav 路径。

    audio-separator 的 Separator 初始化会下载/缓存模型（首次运行需联网）。
    """
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:  # 依赖缺失时给出明确指引
        raise RuntimeError(
            "未安装 audio-separator，请先执行：pip install audio-separator"
        ) from exc

    # audio-separator 0.47.0 新 API：实例化时不传模型/设备，先 load_model 再 separate。
    # 设备由库自动检测（CUDA/CPU）。
    separator = Separator(
        output_dir=str(work_dir),
        output_format=SEPARATOR_OUT_FORMAT,
    )
    separator.load_model(SEPARATOR_MODEL)
    created_files = separator.separate(str(other_path))

    # 注意：separate() 返回的是"裸文件名"（相对 cwd），实际文件写在 work_dir 里，
    # 因此拼接 work_dir 得到完整路径再判断存在性。
    def full_path(name: str) -> Path:
        return work_dir / Path(name).name

    # 从返回列表中挑出"伴奏/无音轨"(Instrumental)，而不是人声轨(Vocals)
    instrumental = None
    for f in created_files:
        low = f.lower()
        if "instrumental" in low or ("no_vocal" in low) or ("no_vocals" in low):
            p = full_path(f)
            if p.exists():
                instrumental = p
                break
    if instrumental is None:
        # 兜底：取不带 vocals 字样的文件
        for f in created_files:
            if "vocal" not in Path(f).stem.lower():
                p = full_path(f)
                if p.exists():
                    instrumental = p
                    break
    if instrumental is None or not instrumental.exists():
        raise RuntimeError(f"audio-separator 未生成伴奏轨。创建文件：{created_files}")

    return instrumental


# ---------------------------------------------------------------------------
# 第三步：pyrubberband 分轨变调
# ---------------------------------------------------------------------------
def pitch_shift_track(path: Path, semitones: int, out_path: Path) -> None:
    """
    对单条音轨做半音变调并写出。变调不影响时长。

    强制使用 rubberband 的 R3 精细引擎（-3）：rubberband 以 "rubberband" 名字调用时
    默认是 R2 快速引擎（-2，低质量、伪影多），R3 几乎总是更干净，代价只是更慢。
    """
    try:
        import pyrubberband as rb
    except ImportError as exc:
        raise RuntimeError("未安装 pyrubberband，请先执行：pip install pyrubberband") from exc

    data, sr = load_audio(path)
    if semitones == 0:
        shifted = data
    else:
        # rbargs 里的键/值会原样拼进 rubberband CLI；"-3" 是单值开关，值传空串
        shifted = rb.pitch_shift(data, sr, n_steps=semitones, rbargs={"-3": ""})
    sf.write(str(out_path), shifted.astype(np.float32), sr)


# ---------------------------------------------------------------------------
# 主流程：分离 -> 清洗 -> 变调 -> 合并
# ---------------------------------------------------------------------------
def process_pipeline(input_path: str, semitones: int, boost_db: float,
                     demucs_model: str = "htdemucs", clean_other: bool = True,
                     shift_mode: str = "merge",
                     status: callable = None):
    """
    执行完整流水线，status 用于回传进度文本（用于 Gradio 界面提示）。
    最终返回合并后伴奏的 .wav 路径（位于 OUTPUT_DIR，不随临时目录删除）。

    shift_mode:
      "merge" —— 先把各乐器轨合并成一条"纯伴奏"再整体变调（一次拉伸，相位更一致，伪影更少，推荐）
      "stems" —— 各分轨分别变调后再合并（保留分轨独立性，但多次独立拉伸会叠加相位伪影）
    """
    if status is None:
        status = print
    input_path = Path(input_path)
    semitones = int(np.clip(semitones, SEMITONE_MIN, SEMITONE_MAX))
    boost_db = float(boost_db)

    # 全程使用临时目录承载中间产物，结束后自动彻底清理
    with tempfile.TemporaryDirectory(prefix="karaoke_") as tmp:
        tmp_dir = Path(tmp)

        # ---- 第一步：粗分 ----
        status(f"第一步 1/4：Demucs({demucs_model}) 分离音轨...")
        stems = demucs_separate(input_path, tmp_dir, model=demucs_model)
        status(f"✅ 粗分完成：{ ' / '.join(stems.keys()) }")

        # ---- 第二步：丢弃 vocals，清洗 other ----
        instrument_stems = {k: v for k, v in stems.items() if k != "vocals"}
        if clean_other and "other" in instrument_stems:
            status("第二步 2/4：丢弃 vocals，对 other 轨进行 MDX-Net 二次去人声清洗...")
            instrument_stems["other"] = clean_other_track(instrument_stems["other"], tmp_dir)
            status(f"✅ other 轨清洗完成（已清除残留和声）")
        else:
            status("第二步 2/4：丢弃 vocals（" +
                   ("跳过二次清洗，保留 other 原样以最大化保真" if not clean_other else "无 other 轨，跳过清洗") + "）")

        # ---- 第三步：变调 ----
        if shift_mode == "merge":
            # 高保真模式：先合成"纯伴奏"，再整体变调一次
            status(f"第三步 3/4：先合并 {len(instrument_stems)} 个乐器轨为纯伴奏，再整体变调 {semitones:+d} 半音...")
            arrays = [load_audio(p)[0] for p in instrument_stems.values()]
            instrumental = align_and_sum(*arrays)
            inst_path = tmp_dir / "instrumental_sum.wav"
            sf.write(str(inst_path), instrumental, OUTPUT_SR, subtype=OUTPUT_SUBTYPE)
            shifted_path = tmp_dir / "instrumental_shifted.wav"
            pitch_shift_track(inst_path, semitones, shifted_path)
            shifted_paths = {"instrumental": shifted_path}
            status("✅ 纯伴奏整体变调完成")
        else:
            # 分轨独立变调：各轨分别变调后再合并
            status(f"第三步 3/4：对 { ' / '.join(instrument_stems.keys()) } 分别变调 {semitones:+d} 半音...")
            shifted_dir = tmp_dir / "shifted"
            shifted_dir.mkdir(exist_ok=True)
            shifted_paths = {}
            for name, path in instrument_stems.items():
                out = shifted_dir / f"{name}_shifted.wav"
                pitch_shift_track(path, semitones, out)
                shifted_paths[name] = out
            status("✅ 全部分轨变调完成")

        # ---- 第四步：合并 + 归一化 + 响度增强 ----
        status(f"第四步 4/4：numpy 对齐合并 {len(shifted_paths)} 个分轨，归一化并做响度增强...")
        arrays = [load_audio(p)[0] for p in shifted_paths.values()]
        combined = align_and_sum(*arrays)
        combined = peak_normalize(combined)
        # 可选音量增强：线性增益 + 软膝限制器，防爆破音/削波
        combined = apply_loudness_boost(combined, boost_db)
        status(f"✅ 合并完成（音量增强 {boost_db:+.0f} dB），正在写出结果...")

        # 最终产物写到持久目录（临时目录随后会被自动清理）
        stamp = time.strftime("%Y%m%d_%H%M%S")
        boost_tag = f"_boost{boost_db:+.0f}db" if boost_db > 0 else ""
        final_name = f"backing_{stamp}_semi{'+' if semitones>=0 else ''}{semitones}{boost_tag}.wav"
        final_path = OUTPUT_DIR / final_name
        # 以 32-bit float 写出，无损保留动态，避免 16-bit 量化损失
        sf.write(str(final_path), combined, OUTPUT_SR, subtype=OUTPUT_SUBTYPE)
        status(f"🎉 全部完成，已输出：{final_name}")

    # 临时目录已清理
    return str(final_path)


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    def run_pipeline(input_path, semitones, boost_db, quality, clean_other, shift_mode, progress=gr.Progress()):
        # 用生成器方式逐步回传状态文本，实现实时进度提示
        status_box = None
        last_status = ""

        def status(text: str):
            nonlocal last_status
            last_status = text
            print(text)

        try:
            # 先初始化状态组件（通过 gr.Info 也给出全局提示）
            gr.Info("开始处理…首次运行会自动下载 Demucs 与 MDX 模型，请耐心等待。")

            status_box = gr.update(value=f"**状态：** {last_status or '启动中…'}")
            yield status_box, None

            status("准备中：校验输入音频…")
            status_box = gr.update(value=f"**状态：** {last_status}")
            yield status_box, None

            demucs_model = DEMUCS_MODELS.get(quality, "htdemucs")
            result_path = process_pipeline(
                input_path, semitones, boost_db,
                demucs_model=demucs_model, clean_other=bool(clean_other),
                shift_mode="merge" if shift_mode == "合并后一次性变调" else "stems",
                status=status,
            )
            status_box = gr.update(value=f"**状态：** {last_status}")
            yield status_box, result_path

        except Exception as exc:
            err = f"❌ 处理失败：{exc}"
            print(err)
            gr.Warning(err)
            status_box = gr.update(value=f"**状态：** {err}")
            yield status_box, None

    with gr.Blocks(title="母带级 K 歌伴奏生成器") as demo:
        gr.Markdown(
            """
# 🎤 母带级 K 歌伴奏生成器

上传原唱 → Demucs 分离（4/6 轨）→ 丢弃人声、可选 MDX 清洗 other → 变调 → 合并输出纯净伴奏。
> 变调使用 rubberband R3 精细引擎；推荐无损源 + 六分轨 + 关二次清洗 + 合并后一次性变调，保真度最高。
> 首次运行会自动下载模型，耗时较长，属正常现象。版权仅限自有 / 授权 / 合理使用音频。
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="上传原唱 (wav / mp3 / flac / m4a 等)",
                    type="filepath",
                    sources=["upload"],
                )
                semitones = gr.Slider(
                    minimum=SEMITONE_MIN,
                    maximum=SEMITONE_MAX,
                    value=0,
                    step=1,
                    label="变调半音数 (负=降调, 正=升调)",
                )
                boost_db = gr.Slider(
                    minimum=-6,
                    maximum=24,
                    value=0,
                    step=1,
                    label="音量增强 (dB，0=不变)",
                    info=">0 时提高响度，内置软膝限制器自动防止削波爆破音",
                )
                quality = gr.Radio(
                    choices=list(DEMUCS_MODELS.keys()),
                    value="六分轨 (htdemucs_6s)",
                    label="分离模式 / 质量",
                    info="六分轨会单独分离钢琴、吉他、贝斯，去人声时对乐器音色损伤更小；4分轨更快但乐器挤在 other 里",
                )
                clean_other = gr.Checkbox(
                    value=False,
                    label="二次人声清洗（去和声）",
                    info="关闭可最大限度保留乐器原声（推荐配无损源）；开启则更彻底去和声但乐器可能略变薄",
                )
                shift_mode = gr.Radio(
                    choices=["合并后一次性变调", "分轨独立变调"],
                    value="合并后一次性变调",
                    label="变调方式",
                    info="合并后一次性变调：先把乐器合成纯伴奏再整体变调一次，相位更一致、伪影更少（推荐）；分轨独立变调会叠加多次拉伸的伪影",
                )
                run_btn = gr.Button("🚀 生成伴奏", variant="primary")
            with gr.Column(scale=1):
                status_md = gr.Markdown("**状态：** 等待上传音频…")
                audio_out = gr.Audio(
                    label="生成结果（纯净变调伴奏）",
                    type="filepath",
                    interactive=False,
                )

        run_btn.click(
            fn=run_pipeline,
            inputs=[audio_in, semitones, boost_db, quality, clean_other, shift_mode],
            outputs=[status_md, audio_out],
        )

    demo.queue()
    return demo


# ---------------------------------------------------------------------------
# 程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="母带级 K 歌伴奏生成器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成公网临时链接")
    args = parser.parse_args()

    print(f"设备: {DEVICE} | 模型: {SEPARATOR_MODEL}")
    print(f"结果输出目录: {OUTPUT_DIR}")
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
