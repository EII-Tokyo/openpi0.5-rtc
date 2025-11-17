#!/usr/bin/env python3
"""
语音助手脚本
功能：
1. 使用VAD检测麦克风输入
2. 语音识别（Whisper）
3. ChatGPT处理指令并生成回复
4. Redis发送操作指令
5. TTS播放回复
"""

import os
import json
import time
import threading
import queue
import uuid
import redis
import openai
import pyaudio
import numpy as np
import torch
import whisper
from io import BytesIO
import pygame
from TTS.api import TTS
import os
from config import OPENAI_API_KEY, REDIS_HOST, REDIS_PORT, REDIS_DB, VAD_THRESHOLD

class VoiceAssistant:
    def __init__(self):
        # 初始化配置 - 优先使用环境变量
        self.openai_api_key = os.getenv('OPENAI_API_KEY', OPENAI_API_KEY)
        self.redis_host = os.getenv('REDIS_HOST', REDIS_HOST)
        self.redis_port = int(os.getenv('REDIS_PORT', REDIS_PORT))
        self.redis_db = int(os.getenv('REDIS_DB', REDIS_DB))

        # Redis pub/sub channels - match buildup_demo.py configuration
        self.instruction_channel = os.getenv('INSTRUCTION_CHANNEL', 'robot_instructions')
        self.status_channel = os.getenv('STATUS_CHANNEL', 'robot_status')

        # Status monitoring
        self.enable_status_monitoring = os.getenv('ENABLE_STATUS_MONITORING', 'false').lower() == 'true'
        self.status_thread = None
        self.status_running = False

        # 初始化组件
        self.setup_audio()
        self.setup_vad()
        self.setup_whisper()
        self.setup_redis()
        self.setup_openai()
        self.setup_tts()
        
        # 音频队列
        self.audio_queue = queue.Queue()
        self.is_recording = False

        # ChatGPT提示词模板 - 用于翻译任何语言到英文
        # Load the prompt template from file
        template_path = os.path.join(os.path.dirname(__file__), 'system_prompt_template.txt')
        with open(template_path, 'r', encoding='utf-8') as f:
            self.chatgpt_prompt_template = f.read()
        
        # Start status monitoring if enabled
        if self.enable_status_monitoring:
            self.start_status_monitoring()

        print("语音助手初始化完成！")
        print(f"Redis连接: {self.redis_host}:{self.redis_port}")
        print(f"发送指令到频道: {self.instruction_channel}")
        print(f"监听状态频道: {self.status_channel}")
        if self.enable_status_monitoring:
            print("状态监控: 已启用")
        print("\n等待语音输入...")

    def setup_audio(self):
        """设置音频设备"""
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.audio = pyaudio.PyAudio()
        
        # 查找Pulse设备
        device_count = self.audio.get_device_count()
        pulse_device = None
        
        for i in range(device_count):
            device_info = self.audio.get_device_info_by_index(i)
            # print(f"设备信息: {device_info}")
            if device_info['maxInputChannels'] > 0 and 'pulse' in device_info['name'].lower():
                pulse_device = i
                print(f"找到Pulse设备: {device_info['name']}")
                break
        
        if pulse_device is None:
            raise RuntimeError("未找到Pulse音频设备")
        
        # 保存设备索引以便重新打开时使用
        self.pulse_device = pulse_device
        
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=pulse_device,
            frames_per_buffer=self.CHUNK
        )
        print(f"Pulse设备初始化成功，采样率: {self.RATE}Hz")

    def setup_vad(self):
        """设置语音活动检测"""
        # 使用Silero VAD模型
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.vad_threshold = VAD_THRESHOLD  # VAD阈值
        print("Silero VAD模型加载完成")

    def setup_whisper(self):
        """设置Whisper模型"""
        print("正在加载Whisper模型...")
        
        # 强制检查CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用！此系统必须使用CUDA。")
        
        print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        self.whisper_model = whisper.load_model("large-v3", device="cuda")
        print("Whisper large-v3模型加载完成")

    def setup_redis(self):
        """设置Redis连接"""
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=True
        )
        self.redis_client.ping()
        print("Redis连接成功")

    def setup_openai(self):
        """设置OpenAI客户端"""
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        print("OpenAI客户端初始化完成")

    def setup_tts(self):
        """设置TTS"""
        pygame.mixer.init()

        # 初始化Coqui TTS
        print("正在加载Coqui TTS模型...")

        # 强制检查CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用！此系统必须使用CUDA。")

        print(f"TTS使用CUDA设备: {torch.cuda.get_device_name(0)}")

        # 修复 PyTorch 2.6+ 的 TTS 加载问题 - 猴子补丁 load_fsspec 函数
        try:
            import TTS.utils.io as tts_io
            original_load_fsspec = tts_io.load_fsspec

            def patched_load_fsspec(path, map_location=None, cache=True, **kwargs):
                """Patched version that disables weights_only for TTS model loading"""
                # Remove weights_only from kwargs if present
                kwargs.pop('weights_only', None)
                # Force weights_only=False for TTS models (they are from a trusted source)
                kwargs['weights_only'] = False
                return original_load_fsspec(path, map_location, cache, **kwargs)

            tts_io.load_fsspec = patched_load_fsspec
            print("已应用TTS加载补丁以兼容PyTorch 2.6+")
        except Exception as e:
            print(f"Warning: Could not patch TTS loading: {e}")

        # 使用多语言TTS模型
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda")
        print("Coqui TTS多语言模型加载完成")

        print("TTS初始化完成")

    def detect_voice_activity(self, audio_data):
        """检测语音活动"""
        # 将音频数据转换为tensor
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        
        # 使用Silero VAD检测语音活动
        speech_prob = self.vad_model(audio_tensor, self.RATE).item()
        return speech_prob > self.vad_threshold

    def record_audio(self):
        """录制音频"""
        print("开始录音...")
        
        # 重新打开音频流以清空缓存
        print("重新打开音频流...")
        self.stream.stop_stream()
        self.stream.close()
        
        # 重新打开音频流
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=self.pulse_device,
            frames_per_buffer=self.CHUNK
        )
        
        frames = []
        silent_frames = 0
        max_silent_frames = 30  # 最多30帧静音后停止录音
        
        while self.is_recording:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # VAD检测
            if self.detect_voice_activity(audio_data):
                frames.append(data)
                silent_frames = 0
            else:
                if frames:  # 如果已经开始录音
                    silent_frames += 1
                    frames.append(data)
                    
                    if silent_frames >= max_silent_frames:
                        break
        
        print("录音结束")
        return b''.join(frames)

    def speech_to_text(self, audio_data):
        """语音转文字"""
        print("正在识别语音...")
        
        # 将音频数据转换为numpy数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 使用Whisper识别，让Whisper自动检测语言
        result = self.whisper_model.transcribe(audio_np)
        text = result["text"].strip()
        detected_language = result["language"]
        
        print(f"识别结果: {text}")
        print(f"Whisper检测到的语言: {detected_language}")
        
        # 只支持英语(en)、中文(zh)和日语(ja)
        supported_languages = ["en", "zh", "ja"]
        if detected_language not in supported_languages:
            print(f"不支持的语言: {detected_language}，只支持: {supported_languages}")
            return None, detected_language
        
        return text, detected_language

    def get_chatgpt_response(self, user_text, detected_language):
        """获取ChatGPT翻译响应"""
        try:
            # 使用初始化中定义的提示词模板
            prompt = self.chatgpt_prompt_template.format(
                user_text=user_text,
                detected_language=detected_language
            )

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a translation assistant for a robot control system. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=200,
            )

            response_text = response.choices[0].message.content.strip()
            print(f"ChatGPT响应: {response_text}")

            # 清理响应文本，移除markdown代码块标记
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # 移除 ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]   # 移除 ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]   # 移除结尾的 ```
            response_text = response_text.strip()

            # 解析JSON响应
            try:
                response_json = json.loads(response_text)
                english_instruction = response_json.get("english_instruction")
                reply_text = response_json.get("response_statement", "")

                return english_instruction, reply_text
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print(f"清理后的响应: {response_text}")
                return None, None

        except Exception as e:
            print(f"ChatGPT请求失败: {e}")
            return None, None

    def send_to_redis(self, english_instruction):
        """发送指令到Redis使用pub/sub模式

        发送英文指令到机器人控制系统
        """
        # Check if it's a stop command
        if "stop" in english_instruction.lower():
            message = {"command": "stop"}
            print(f"发送停止命令: {english_instruction}")
        else:
            # Send as instruction
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            message = {
                "instruction": english_instruction,
                "task_id": task_id,
            }
            print(f"发送任务指令 {task_id}: {english_instruction}")

        # 使用pub/sub模式发布消息
        self.redis_client.publish(self.instruction_channel, json.dumps(message))
        print(f"消息已发送到频道: {self.instruction_channel}")

    def text_to_speech(self, text, detected_language=None):
        """文字转语音"""
        print(f"正在播放: {text}")
        
        # 生成临时音频文件
        temp_file = "/tmp/tts_output.wav"
        
        # 使用指定的TTS参数
        self.tts.tts_to_file(
            text=text,
            file_path=temp_file,
            speaker="Alexandra Hisakawa",
            language=detected_language,
            split_sentences=True
        )
        
        # 使用pygame播放
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        # 等待播放完成
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def process_voice_command(self):
        """处理语音指令"""
        # 开始录音
        self.is_recording = True
        audio_data = self.record_audio()
        self.is_recording = False

        if len(audio_data) == 0:
            print("没有检测到语音")
            return

        # 语音转文字
        text, detected_language = self.speech_to_text(audio_data)
        if text is None:
            print("不支持的语言")
            self.text_to_speech("申し訳ございませんが、対応していない言語です。英語、中国語、または日本語でお話しください。", "ja")
            return
        elif not text:
            print("没有检测到语音")
            self.text_to_speech("申し訳ございませんが、よく聞き取れませんでした。もう一度お話しください。", "ja")
            return

        # ChatGPT翻译成英文
        english_instruction, reply_text = self.get_chatgpt_response(text, detected_language)

        # 打印检测到的语言
        print(f"检测到的语言: {detected_language}")
        print(f"英文指令: {english_instruction}")

        # 发送英文指令到Redis
        if english_instruction:
            self.send_to_redis(english_instruction)
            self.text_to_speech(reply_text, detected_language)
        else:
            self.text_to_speech("申し訳ございませんが、お話の内容を理解できませんでした。もう一度お話しください。", "ja")

    def run(self):
        """主运行循环"""
        try:
            while True:
                print("\n控制方式:")
                print("  - 按Enter键开始语音识别")
                print("  - 输入'quit'退出")
                print("等待输入...")

                user_input = input().strip()

                if user_input.lower() == 'quit':
                    break
                elif user_input == '':
                    self.process_voice_command()
                else:
                    print("无效输入，请按Enter进行语音识别")

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.cleanup()

    def start_status_monitoring(self):
        """启动状态监控线程"""
        if self.status_running:
            return

        self.status_running = True
        self.status_thread = threading.Thread(target=self._status_listener, daemon=True)
        self.status_thread.start()
        print("状态监控线程已启动")

    def stop_status_monitoring(self):
        """停止状态监控线程"""
        self.status_running = False
        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=2.0)
        print("状态监控线程已停止")

    def _status_listener(self):
        """监听机器人状态更新"""
        # Create a separate Redis connection for the pubsub
        pubsub_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=True
        )
        pubsub = pubsub_client.pubsub()
        pubsub.subscribe(self.status_channel)

        print(f"开始监听状态频道: {self.status_channel}")

        try:
            while self.status_running:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        task_id = data.get('task_id', 'unknown')
                        status = data.get('status', 'unknown')

                        # Format status message for display
                        status_msg = f"[状态更新] 任务 {task_id}: {status}"

                        if 'instruction' in data:
                            status_msg += f" - {data['instruction']}"
                        if 'progress' in data:
                            progress = data['progress']
                            status_msg += f" ({progress['current_step']}/{progress['max_steps']})"
                        if 'error' in data:
                            status_msg += f" - 错误: {data['error']}"

                        print(f"\n{status_msg}")

                    except json.JSONDecodeError as e:
                        print(f"状态消息JSON解析失败: {e}")
        except Exception as e:
            print(f"状态监听线程异常: {e}")
        finally:
            pubsub.close()
            pubsub_client.close()
            print("状态监听线程结束")

    def cleanup(self):
        """清理资源"""
        # Stop status monitoring
        if self.enable_status_monitoring:
            self.stop_status_monitoring()

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        pygame.mixer.quit()
        print("资源清理完成")

def main():
    """主函数"""
    assistant = VoiceAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
