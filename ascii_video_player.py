import cv2
import os
import time
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime

# -----------------------------
# ASCII 문자 변환 함수 정의
# -----------------------------
def pixel_to_ascii(pixel_intensity):
    ASCII_CHARS = "   ._-=+*!&#%$@"
    return ASCII_CHARS[int(pixel_intensity) * len(ASCII_CHARS) // 256]

def pixel_to_ascii_color(b, g, r):
    ASCII_CHARS = "@%#*+=-:. "
    brightness = int((r + g + b) / 3)
    char = ASCII_CHARS[brightness * len(ASCII_CHARS) // 256]
    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

# -----------------------------
# 흑백 ASCII 이미지를 Pillow로 렌더링
# -----------------------------
def ascii_to_image(ascii_text, font_path="C:/Windows/Fonts/consola.ttf", font_size=10):
    lines = ascii_text.split("\n")
    max_width = max(len(line) for line in lines)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    img = Image.new("RGB", (char_width * max_width, char_height * len(lines)), "black")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((0, i * char_height), line, fill=(255, 255, 255), font=font)
    return img

# -----------------------------
# 컬러 ASCII 이미지를 Pillow로 렌더링
# -----------------------------
def ascii_to_image_colored(frame, width, font_path="C:/Windows/Fonts/consola.ttf", font_size=10):
    h, w = frame.shape[:2]
    aspect_ratio = 0.4194
    height = int((width * h / w) * aspect_ratio)
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    img = Image.new("RGB", (char_width * width, char_height * height), "black")
    draw = ImageDraw.Draw(img)

    for i in range(height):
        line = ""
        colors = []
        for j in range(width):
            b, g, r = resized[i, j]
            brightness = int((r + g + b) / 3)
            char = "@%#*+=-:. "[brightness * 10 // 256]
            line += char
            colors.append((r, g, b))
        for j, ch in enumerate(line):
            draw.text((j * char_width, i * char_height), ch, fill=colors[j], font=font)

    return img

# -----------------------------
# 프레임을 ASCII 문자열로 변환
# -----------------------------
def frame_to_ascii(frame, width, use_color):
    h, w = frame.shape[:2]
    aspect_ratio = 0.4194
    height = int((width * h / w) * aspect_ratio)

    if use_color:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        ascii_lines = []
        for i in range(height):
            line = []
            for j in range(width):
                b, g, r = resized[i, j]
                line.append(pixel_to_ascii_color(b, g, r))
            ascii_lines.append(''.join(line))
        return '\n'.join(ascii_lines)

    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
        ascii_lines = []
        for i in range(height):
            line = []
            for j in range(width):
                line.append(pixel_to_ascii(resized[i, j]))
            ascii_lines.append(''.join(line))
        return '\n'.join(ascii_lines)


# -----------------------------
# 비디오 재생 및 키 조작 처리
# -----------------------------
def play_video_ascii(video_path, init_width=150):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 0.033

    use_color = False
    width = init_width
    terminal_max_width = shutil.get_terminal_size((150, 40)).columns
    save_video = False
    saved_frames = []

    cv2.namedWindow("dummy", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        ascii_frame = frame_to_ascii(frame, width, use_color)
        print("\033[H", end="")  # 터미널 커서 이동
        print(ascii_frame)

        if save_video:
            img = ascii_to_image_colored(frame, width) if use_color else ascii_to_image(ascii_frame)
            saved_frames.append(np.array(img))

        cv2.imshow("dummy", frame[0:1, 0:1])  # 키 입력 감지를 위한 더미 창

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            use_color = not use_color
        elif key == ord('q'):
            break
        elif key == ord('-'):
            if save_video:
                print("[WARN] Cannot change resolution during recording.")
            else:
                os.system('cls' if os.name == 'nt' else 'clear')
                width = max(20, width - 10)
        elif key == ord('+') or key == ord('='):
            if save_video:
                print("[WARN] Cannot change resolution during recording.")
            else:
                os.system('cls' if os.name == 'nt' else 'clear')
                width = min(terminal_max_width, width + 10)
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"text/ascii_frame_{timestamp}q.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(ascii_frame)
        elif key == ord('v'):
            save_video = not save_video
            print("[INFO] Video recording {}".format("started" if save_video else "stopped"))

        time.sleep(frame_delay)

    if saved_frames:
        frame_h, frame_w, _ = saved_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        color_mode = "color" if use_color else "gray"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording/{color_mode}_{timestamp}.avi"
        out = cv2.VideoWriter(filename, fourcc, fps, (frame_w, frame_h))
        for frame in saved_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[INFO] Saved {filename}")

    video.release()
    cv2.destroyAllWindows()

# -----------------------------
# 메인 실행
# -----------------------------
if __name__ == "__main__":
    video_path = "source_video/sample1.mp4"  # 재생할 비디오 경로 설정
    os.system('cls' if os.name == 'nt' else 'clear')
    play_video_ascii(video_path)
