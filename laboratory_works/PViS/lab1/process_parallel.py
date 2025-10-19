import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np


def process_range(
    input_path,
    output_path,
    lower_rgb,
    upper_rgb,
    min_area,
    kernel_size,
    start_frame,
    end_frame,
):
    cap = cv.VideoCapture(input_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mask = cv.inRange(rgb, np.array(lower_rgb), np.array(upper_rgb))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask)

        region_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n_labels):
            x, y, w_box, h_box, area = stats[i]
            if area >= min_area:
                cv.rectangle(region_mask, (x, y), (x + w_box, y + h_box), 255, -1)

        blurred = cv.filter2D(frame, -1, kernel)
        region_mask_3ch = cv.merge([region_mask, region_mask, region_mask])
        result = np.where(region_mask_3ch == 255, frame, blurred)

        out.write(result)

    cap.release()
    out.release()


def process_video_parallel(
    input_path, output_path, lower_rgb, upper_rgb, min_area, kernel_size, num_processes
):
    cap = cv.VideoCapture(input_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.release()

    frames_per_part = frame_count // num_processes
    temp_files = []

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(num_processes):
            start = i * frames_per_part
            end = frame_count if i == num_processes - 1 else (i + 1) * frames_per_part
            temp_file = f"output/temp_part_{i}.mp4"
            temp_files.append(temp_file)
            futures.append(
                executor.submit(
                    process_range,
                    input_path,
                    temp_file,
                    lower_rgb,
                    upper_rgb,
                    min_area,
                    kernel_size,
                    start,
                    end,
                )
            )

        for f in futures:
            f.result()

    elapsed = time.perf_counter() - start_time
    print(f"\nВремя обработки ({num_processes} процессов): {elapsed:.2f} c")

    # объединяем видео
    concat_videos(temp_files, output_path, fps)

    # удаляем временные куски
    for f in temp_files:
        Path(f).unlink(missing_ok=True)


def concat_videos(temp_files, output_path, fps):
    """Объединение временных видео в одно с помощью OpenCV"""
    first = cv.VideoCapture(temp_files[0])
    w = int(first.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(first.get(cv.CAP_PROP_FRAME_HEIGHT))
    first.release()

    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for part in temp_files:
        cap = cv.VideoCapture(part)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"Собрано итоговое видео: {output_path}")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    process_video_parallel(
        input_path="input/orange_1080p_120s.mp4",
        output_path="output/orange_1080p_120s_parallel.mp4",
        lower_rgb=(214, 117, 13),
        upper_rgb=(255, 197, 93),
        min_area=500,
        kernel_size=5,
        num_processes=4,  # можно менять 1, 2, 4, 6, 8, 10
    )
