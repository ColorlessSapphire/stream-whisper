#!/usr/bin/env python
"""
在本地进行录音 + 转写的单脚本代码。不依赖于云服务（e.g., redis, socket），适合于离线使用。

依赖安装:
    pip3 install pyaudio webrtcvad faster-whisper

运行方式:
    python3 local_deploy.py
"""

import collections
import io
import logging
import queue
import threading
import time
import typing
import wave
import warnings
from io import BytesIO
import numpy as np
import torch
import scipy.signal as signal

import pyaudio
import webrtcvad
from faster_whisper import WhisperModel
import whisper

# 忽略Whisper的警告
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')

# Whisper模型需要的采样率
WHISPER_SAMPLE_RATE = 16000

# 需要去除的短语列表
UNWANTED_PHRASES = [
    "转写正确",
    "中文字幕",
    "字幕组",
    "字幕制作",
    "感谢观看",
    "点赞订阅",
    "关注我们",
    "更多精彩",
    "电台广告",
    "转写广播电台",
    "简体中文",
    "普通话",
    "忽略音乐",
    "音乐音乐"
]

# 需要检测的关键词列表
KEYWORDS = [
    "交通",
    "天气",
    "新闻",
    "广告",
    "音乐",
    "节目预告",
    "路况",
    "预警",
    "事故",
    "直播",
]

class MultiChannelQueues:
    """多通道音频和文本队列管理器"""
    def __init__(self, num_channels: int = 2):
        self.audio_queues = {i: queue.Queue() for i in range(num_channels)}
        self.text_queues = {i: queue.Queue() for i in range(num_channels)}
        self.raw_audio_queues = {i: queue.Queue() for i in range(num_channels)}
        self.num_channels = num_channels
        self.transcription_complete = {i: threading.Event() for i in range(num_channels)}
        for event in self.transcription_complete.values():
            event.set()

    def wait_all_transcription_complete(self):
        for channel in range(self.num_channels):
            self.transcription_complete[channel].wait()

    def reset_transcription_flags(self):
        for channel in range(self.num_channels):
            self.transcription_complete[channel].clear()

def resample_audio(audio_data: bytes, src_sample_rate: int, target_sample_rate: int, sample_width: int = 2) -> np.ndarray:
    """重采样音频数据"""
    # 将字节数据转换为numpy数组
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    
    # 计算重采样后的长度
    output_length = int(len(audio_np) * target_sample_rate / src_sample_rate)
    
    # 使用scipy进行重采样
    resampled = signal.resample(audio_np, output_length)
    
    # 确保数据类型正确
    resampled = np.clip(resampled, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    resampled = resampled.astype(np.int16)
    
    return resampled

class AudioPlayer(threading.Thread):
    """音频播放器类，用于实时播放指定通道的音频"""
    def __init__(self, queues: MultiChannelQueues, channel: int, 
                 sample_rate: int = 16000, sample_width: int = 2):
        super().__init__()
        self.queues = queues
        self.channel = channel
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.daemon = True
        self.stop_flag = threading.Event()
        
        logging.info(f"初始化音频播放器: 通道 {channel}, 采样率 {sample_rate}Hz, 采样宽度 {sample_width}字节")
        
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=self.audio.get_format_from_width(sample_width),
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=1024
            )
            logging.info("音频播放器初始化成功")
            
        except Exception as e:
            logging.error(f"音频播放器初始化失败: {str(e)}")
            raise
        
    def run(self):
        try:
            while not self.stop_flag.is_set():
                try:
                    audio_data = self.queues.raw_audio_queues[self.channel].get(timeout=0.1)
                    if not audio_data or len(audio_data) == 0:
                        continue
                    
                    self.stream.write(audio_data)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"音频播放错误: {str(e)}")
                    continue
                    
        finally:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()
                logging.info("音频播放器已关闭")
            except Exception as e:
                logging.error(f"关闭音频播放器时出错: {str(e)}")
            
    def stop(self):
        self.stop_flag.set()

class OpenAITranscriber(threading.Thread):
    """使用 OpenAI Whisper 模型的转录器类"""
    def __init__(
        self,
        model_size: str = 'base',
        device: str = 'cuda' if torch.cuda.is_available() else "cpu",
        language: str = 'zh',
        task: str = "transcribe",
        initial_prompt: str = '以下是普通话的句子，这是一段广播电台节目：',
        channel: int = 0,
        queues: MultiChannelQueues = None,
        unwanted_phrases: list = None,
        keywords: list = None) -> None:
        
        super().__init__()
        self.model_size = model_size
        self.device = device
        self.language = language
        self.task = task
        self.initial_prompt = initial_prompt
        self.channel = channel
        self.queues = queues
        self.daemon = True
        
        # 使用默认列表或自定义列表
        self.unwanted_phrases = unwanted_phrases or UNWANTED_PHRASES
        self.keywords = keywords or KEYWORDS
        
        logging.info(f"正在加载 Whisper 模型 {model_size}...")
        self._model = whisper.load_model(self.model_size, device=self.device)
        logging.info("Whisper 模型加载完成")

    def detect_keywords(self, text: str) -> list:
        """检测文本中的关键词"""
        found_keywords = []
        for keyword in self.keywords:
            if keyword in text:
                found_keywords.append(keyword)
        return found_keywords

    def filter_text(self, text: str) -> str:
        """过滤和清理转写文本"""
        # 移除不需要的短语
        filtered_text = text.strip()
        for phrase in self.unwanted_phrases:
            filtered_text = filtered_text.replace(phrase, "")
        
        # 检查重复内容
        words = filtered_text.split()
        if len(words) >= 3:
            # 检查连续重复的短语
            phrase = ' '.join(words[:3])
            count = filtered_text.count(phrase)
            if count > 2:  # 如果一个短语重复出现超过2次，可能是错误结果
                return ''
            
            # 检查交替重复的短语
            half_len = len(words) // 2
            if half_len >= 3:
                first_half = ' '.join(words[:half_len])
                second_half = ' '.join(words[half_len:2*half_len])
                if first_half == second_half:
                    return first_half
        
        # 清理多余的空白字符
        filtered_text = ' '.join(filtered_text.split())
        return filtered_text

    def transcribe_audio(self, audio: bytes, src_sample_rate: int) -> typing.Generator[tuple[str, list], None, None]:
        """转写音频数据，返回转写文本和检测到的关键词"""
        try:
            # 转换音频数据为numpy数组
            audio_np = np.frombuffer(audio, dtype=np.int16)
            
            # 如果源采样率不是16kHz，进行重采样
            if src_sample_rate != WHISPER_SAMPLE_RATE:
                audio_np = resample_audio(audio, src_sample_rate, WHISPER_SAMPLE_RATE)
            
            # 将音频数据标准化到[-1, 1]范围
            audio_np = audio_np.astype(np.float32) / 32768.0
            
            # 使用Whisper进行转写
            result = self._model.transcribe(
                audio_np,
                temperature=0,
                language=self.language,
                task=self.task,
                initial_prompt=self.initial_prompt,
            )
            
            # 处理转写结果
            for segment in result["segments"]:
                text = segment["text"].strip()
                filtered_text = self.filter_text(text)
                if filtered_text:
                    # 检测关键词
                    keywords = self.detect_keywords(filtered_text)
                    yield filtered_text, keywords
                    
        except Exception as e:
            logging.error(f"转写过程出错: {str(e)}")

    def run(self):
        while True:
            if not self.queues:
                logging.error("No queues provided for transcriber")
                break
                
            try:
                logging.debug(f"通道 {self.channel} 等待音频数据...")
                audio_data = self.queues.audio_queues[self.channel].get()
                logging.debug(f"通道 {self.channel} 开始转写...")
                
                text_segments = []
                with wave.open(BytesIO(audio_data), 'rb') as wf:
                    sample_rate = wf.getframerate()
                    audio_bytes = wf.readframes(wf.getnframes())
                    
                    for text, keywords in self.transcribe_audio(audio_bytes, sample_rate):
                        text_segments.append(text)
                        if keywords:
                            logging.info(f"Channel {self.channel} 检测到关键词: {', '.join(keywords)}")
                
                if text_segments:
                    full_text = " ".join(text_segments)
                    if full_text.strip():
                        self.queues.text_queues[self.channel].put(full_text)
                
                self.queues.transcription_complete[self.channel].set()
                
            except Exception as e:
                logging.error(f"Channel {self.channel} transcription error: {e}")
                self.queues.transcription_complete[self.channel].set()

class AudioFileReader(threading.Thread):
    """定时读取音频文件并送入队列处理"""
    def __init__(self, 
                 file_path: str, 
                 queues: MultiChannelQueues, 
                 chunk_duration: float = 30.0) -> None:
        super().__init__()
        self.file_path = file_path
        self.queues = queues
        self.chunk_duration = chunk_duration
        self.daemon = True
        
        with wave.open(self.file_path, 'rb') as wf:
            self.channels = wf.getnchannels()
            self.sample_width = wf.getsampwidth()
            self.framerate = wf.getframerate()
            self.nframes = wf.getnframes()
            self.chunk_size = int(self.framerate * chunk_duration)
            logging.info(f"音频文件信息: {self.channels}声道, {self.framerate}Hz, {self.sample_width}字节")
    
    def create_wave_file(self, audio_data: bytes, sample_width: int, framerate: int) -> bytes:
        """创建WAVE文件"""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(sample_width)
            out_wf.setframerate(framerate)
            out_wf.writeframes(audio_data)
        return buf.getvalue()
    
    def read_audio_chunk(self, track: int, start_frame: int) -> tuple[bytes, bool]:
        try:
            with wave.open(self.file_path, 'rb') as wf:
                if track >= wf.getnchannels():
                    logging.error(f"音轨 {track} 不存在，文件只有 {wf.getnchannels()} 个音轨")
                    return b'', False, b''
                
                wf.setpos(start_frame)
                frames_to_read = min(self.chunk_size, self.nframes - start_frame)
                if frames_to_read <= 0:
                    return b'', False, b''
                    
                frames = wf.readframes(frames_to_read)
                
                if self.channels > 1:
                    samples = np.frombuffer(frames, dtype=np.int16)
                    samples = samples.reshape(-1, self.channels)
                    channel_data = samples[:, track]
                    frames = channel_data.tobytes()
                
                # 创建用于播放的原始音频数据
                raw_audio = frames
                
                # 创建用于转写的WAVE文件
                wave_audio = self.create_wave_file(frames, self.sample_width, self.framerate)
                
                has_more = (start_frame + frames_to_read) < self.nframes
                return wave_audio, has_more, raw_audio
                
        except Exception as e:
            logging.error(f"读取音频文件时出错: {e}")
            return b'', False, b''
    
    def run(self):
        current_frame = 0
        
        while True:
            try:
                logging.debug("等待所有通道转写完成...")
                self.queues.wait_all_transcription_complete()
                logging.debug("所有通道转写完成，开始读取下一段音频...")
                
                self.queues.reset_transcription_flags()
                
                all_channels_complete = True
                for channel in range(min(self.channels, self.queues.num_channels)):
                    audio_data, has_more, raw_audio = self.read_audio_chunk(channel, current_frame)
                    if audio_data:
                        self.queues.audio_queues[channel].put(audio_data)
                        self.queues.raw_audio_queues[channel].put(raw_audio)
                        logging.debug(f"已读取音频文件通道 {channel} 的数据片段")
                        
                        if has_more:
                            all_channels_complete = False
                
                current_frame += self.chunk_size
                
                if all_channels_complete:
                    logging.info("所有音频数据处理完成")
                    break
                    
            except Exception as e:
                logging.error(f"处理音频文件时出错: {e}")
                break

def main():
    try:
        audio_file = input("请输入音频文件路径: ")
        num_channels = int(input("请输入需要处理的通道数 (默认2): ") or "2")
        chunk_duration = float(input("请输入每个音频片段的时长(秒) (默认30): ") or "30")
        
        queues = MultiChannelQueues(num_channels)
        reader = AudioFileReader(audio_file, queues, chunk_duration)
        
        with wave.open(audio_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
        
        player = AudioPlayer(queues, 0, sample_rate, sample_width)
        player.start()
        
        transcribers = []
        for channel in range(num_channels):
            transcriber = OpenAITranscriber(
                model_size="turbo",
                channel=channel,
                queues=queues
            )
            transcriber.start()
            transcribers.append(transcriber)
        
        reader.start()
            
        try:
            while reader.is_alive() or any(t.is_alive() for t in transcribers):
                for channel in range(num_channels):
                    try:
                        text = queues.text_queues[channel].get_nowait()
                        print(f"Channel {channel}: {text}")
                    except queue.Empty:
                        continue
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n程序终止...")
            player.stop()
            
    except Exception as e:
        logging.error(e, exc_info=True, stack_info=True)

if __name__ == "__main__":
    main()
