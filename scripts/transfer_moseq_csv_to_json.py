# load moseq csv directly to labelv to check the moseq syllable

import pandas as pd
import json
import sys
import io
import os
import cv2
import time

# 解决 Windows 控制台中文输出问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
the first column of csv file is the syllable, each row is one frame.
and only the first column is used.

the json dict:
    id: nan or generate auto
    name: the syllable
    start_frame: the frame number that syllable changed from last syllable
    end_frame: the frame number that syllable changed
    color: the color of the syllable, non
    category: the category of the syllable, default
    description: the description of the syllable, empty
"""

def load_moseq_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

def get_video_metadata(video_path):
    """
    读取视频的元数据信息
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        metadata: 包含视频元数据的字典
    """
    if not os.path.exists(video_path):
        print(f"警告: 视频文件不存在: {video_path}")
        print("使用默认视频参数")
        return {
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 0,
            "duration": 0.0
        }
    
    try:
        # 使用 OpenCV 读取视频信息
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("无法打开视频文件")
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        
        cap.release()
        
        metadata = {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration": duration
        }
        
        print(f"\n视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {frame_count}")
        print(f"  时长: {format_time(duration)}")
        
        return metadata
        
    except Exception as e:
        print(f"读取视频信息失败: {e}")
        print("使用默认视频参数")
        return {
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 0,
            "duration": 0.0
        }

def format_time(seconds):
    """格式化时间为 HH:MM:SS 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def transfer_moseq_csv_to_json(video_path, csv_path, output_name):
    """
    将 MoSeq CSV 文件转换为 LabelV JSON 格式
    
    Args:
        video_path: 视频文件路径
        csv_path: CSV 文件路径
        output_name: 输出 JSON 文件名
    
    Returns:
        export_data: 包含元数据和标签的完整数据
    """
    # 获取视频元数据
    video_metadata = get_video_metadata(video_path)
    fps = video_metadata["fps"]
    
    # 加载 CSV 数据
    df = load_moseq_csv(csv_path)
    
    # 获取第一列 syllable 数据
    syllable_column = df.iloc[:, 0]
    
    # 存储所有标签
    labels = []
    
    # 遍历 syllable 列，检测变化
    current_syllable = syllable_column.iloc[0]
    start_frame = 0
    
    # 为每个 syllable 分配不同的颜色
    colors = [
        [255, 165, 0, 180],   # 橙色
        [0, 191, 255, 180],   # 天蓝色
        [255, 99, 71, 180],   # 番茄红
        [50, 205, 50, 180],   # 绿色
        [138, 43, 226, 180],  # 紫罗兰
        [255, 215, 0, 180],   # 金色
        [255, 20, 147, 180],  # 深粉色
        [64, 224, 208, 180],  # 青绿色
        [255, 140, 0, 180],   # 深橙色
        [147, 112, 219, 180], # 中紫色
    ]
    
    for i in range(1, len(syllable_column)):
        syllable = syllable_column.iloc[i]
        
        # 检测到 syllable 变化
        if syllable != current_syllable:
            # 保存上一个 syllable 段
            label = {
                "id": len(labels) + 1,
                "name": f"syllable_{int(current_syllable)}",
                "start_frame": int(start_frame),
                "end_frame": int(i - 1),
                "color": colors[int(current_syllable) % len(colors)],
                "category": "moseq",
                "description": f"MoSeq syllable {int(current_syllable)}"
            }
            labels.append(label)
            
            # 开始新的 syllable 段
            current_syllable = syllable
            start_frame = i
    
    # 保存最后一个 syllable 段
    label = {
        "id": len(labels) + 1,
        "name": f"syllable_{int(current_syllable)}",
        "start_frame": int(start_frame),
        "end_frame": int(len(syllable_column) - 1),
        "color": colors[int(current_syllable) % len(colors)],
        "category": "moseq",
        "description": f"MoSeq syllable {int(current_syllable)}"
    }
    labels.append(label)
    
    # 创建符合 LabelV 格式的完整数据结构
    total_frames = len(syllable_column)
    duration_sec = total_frames / fps if fps > 0 else 0.0
    
    export_data = {
        "video_file": os.path.basename(video_path),
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration": format_time(duration_sec),
        "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labels": labels
    }
    
    # 保存为 JSON 文件
    output_path = output_name if output_name.endswith('.json') else f"{output_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成！")
    print(f"CSV 帧数: {len(syllable_column)}")
    print(f"视频帧数: {video_metadata.get('frame_count', 'N/A')}")
    print(f"标签数量: {len(labels)}")
    print(f"输出文件: {output_path}")
    
    # 检查帧数是否匹配
    if video_metadata.get('frame_count', 0) > 0:
        if abs(video_metadata['frame_count'] - len(syllable_column)) > 10:
            print(f"\n⚠️ 警告: CSV 帧数与视频帧数差异较大!")
            print(f"   CSV: {len(syllable_column)} 帧")
            print(f"   视频: {video_metadata['frame_count']} 帧")
    
    return export_data

if __name__ == "__main__":
    # 示例用法
    video_path = r"tmp\spose_2025-09-02_14-02-44.mp4"
    csv_path = r"tmp\spose_2025-09-02_14-02-44.csv"
    output_name = "spose_2025-09-02_14-02-44.json"          # 要与视频同名, 放在 tmp 文件夹下
    
    print("=" * 60)
    print("MoSeq CSV 转 LabelV JSON 工具")
    print("=" * 60)
    
    # 执行转换
    export_data = transfer_moseq_csv_to_json(video_path, csv_path, output_name)
    
    # 打印前几个标签示例
    print("\n前 5 个标签示例:")
    for label in export_data['labels'][:5]:
        print(f"  {label['name']}: 帧 {label['start_frame']} - {label['end_frame']}")
    
    print("\n" + "=" * 60)
    print("✓ 转换成功！可以在 LabelV 中加载此 JSON 文件")
