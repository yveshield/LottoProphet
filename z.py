import os
import subprocess
import tempfile
from pdf2image import convert_from_path

def get_mp3_duration(mp3_file):
    """获取 MP3 时长（单位：秒）"""
    result = subprocess.run(
        ['ffmpeg', '-i', mp3_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    for line in result.stderr.splitlines():
        if "Duration" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0

def convert_epub_to_pdf(epub_file, pdf_file):
    """用 ebook-convert 把 EPUB 转为 PDF"""
    subprocess.run([
        "ebook-convert", epub_file, pdf_file,
        "--custom-size", "1280x720",
        "--margin-top", "0", "--margin-bottom", "0",
        "--margin-left", "0", "--margin-right", "0"
    ], check=True)

def convert_pdf_to_images(pdf_file, image_dir):
    """将 PDF 每页转为 PNG 图像"""
    pages = convert_from_path(pdf_file, dpi=150, size=(1280, 720))
    os.makedirs(image_dir, exist_ok=True)
    images = []
    for idx, page in enumerate(pages):
        img_path = os.path.join(image_dir, f"page{idx:04}.png")
        page.save(img_path, "PNG")
        images.append(img_path)
    return images

def generate_video_segments(images, duration_per_image, segment_dir):
    """为每张图片生成一个视频段"""
    os.makedirs(segment_dir, exist_ok=True)
    video_segments = []
    for idx, img in enumerate(images):
        output = os.path.join(segment_dir, f"segment{idx:04}.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1", "-i", img,
            "-t", f"{duration_per_image:.2f}",
            "-vf", "scale=1280:720",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an", output
        ], check=True)
        video_segments.append(output)
    return video_segments

def concat_video_segments(segments, output_file):
    """用 concat 方式合并视频段"""
    list_file = os.path.join(os.path.dirname(segments[0]), "concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(f"file '{segment}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy", output_file
    ], check=True)

def add_audio_to_video(video_file, audio_file, output_file):
    """合并视频和音频"""
    subprocess.run([
        "ffmpeg", "-y", "-i", video_file, "-i", audio_file,
        "-c:v", "copy", "-c:a", "aac", "-shortest", output_file
    ], check=True)

def main(epub_file, mp3_file, output_video):
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir)
        pdf_path = os.path.join(tmpdir, "book.pdf")
        image_dir = os.path.join(tmpdir, "images")
        segment_dir = os.path.join(tmpdir, "segments")
        merged_video = os.path.join(tmpdir, "merged.mp4")

        convert_epub_to_pdf(epub_file, pdf_path)
        images = convert_pdf_to_images(pdf_path, image_dir)

        total_duration = get_mp3_duration(mp3_file)
        duration_per_image = total_duration / len(images)
        print(f"总时长 {total_duration:.2f} 秒，每页显示 {duration_per_image:.2f} 秒")

        segments = generate_video_segments(images, duration_per_image, segment_dir)
        concat_video_segments(segments, merged_video)
        add_audio_to_video(merged_video, mp3_file, output_video)
        print(f"已生成最终视频：{output_video}")

if __name__ == "__main__":
    epub_file = r"G:\Files\Reader\B1+ Intermediate\Agnes_Grey-Anne_Bronte.epub"
    mp3_file = r"G:\Files\Reader\B1+ Intermediate\Agnes_Grey-Anne_Bronte.mp3"
    output_video = "Agnes_Grey_Final.mp4"

    main(epub_file, mp3_file, output_video)
