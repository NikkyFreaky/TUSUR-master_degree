import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

# ======================= Видеопараметры ==========================
VIDEOS = [
    ("input/orange_1080p_30s.mp4", 120),
]

RGB_RANGE = ((214, 117, 13), (255, 197, 93))
MIN_AREA = 500


# ======================= Вспомогательные функции ==========================


def process_range(
    input_path,
    output_path,
    lower_rgb,
    upper_rgb,
    min_area,
    kernel_size,
    start_frame,
    end_frame,
    part_index,
):
    """Обработка части кадров видео — вызывается каждым процессом"""
    cap = cv.VideoCapture(input_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

    # Разные цвета для разных процессов
    COLORS = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
        (128, 255, 0),
    ]
    color = COLORS[part_index % len(COLORS)]
    thickness = 2 + part_index % 3  # чуть разная толщина рамки

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()  # чистый кадр для вставки внутрь рамок
        rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB)

        # бинарная маска по цвету
        mask = cv.inRange(rgb, np.array(lower_rgb), np.array(upper_rgb))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask)

        # соберём прямоугольную маску областей, которые должны остаться РЕЗКИМИ
        region_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n_labels):
            x, y, w_box, h_box, area = stats[i]
            if area >= min_area:
                cv.rectangle(region_mask, (x, y), (x + w_box, y + h_box), 255, -1)

        # (опционально) снимаем «ореол» по границе — чуть расширим маску
        region_mask = cv.dilate(region_mask, np.ones((3, 3), np.uint8), iterations=1)

        # делаем размытое ИЗ исходника (без рамок/текста)
        blurred = cv.filter2D(original, -1, kernel)

        # соберём результирующий кадр: размытый фон + вставка резких областей по маске
        result = blurred.copy()

        # рисуем рамки/подписи на копии оригинала (они НЕ участвуют в размытии)
        annotated = original.copy()
        for i in range(1, n_labels):
            x, y, w_box, h_box, area = stats[i]
            if area >= min_area:
                cv.rectangle(
                    annotated, (x, y), (x + w_box, y + h_box), color, thickness
                )
                cv.putText(
                    annotated,
                    f"P{part_index}",
                    (x + 5, y + 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv.LINE_AA,
                )

        # твёрдая вставка: где маска=255, кладём «annotated» поверх result
        cv.copyTo(annotated, region_mask, result)

        out.write(result)

    cap.release()
    out.release()


def concat_videos(temp_files, output_path, fps):
    """Склейка временных видео"""
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


def process_video_parallel(
    input_path, output_path, lower_rgb, upper_rgb, min_area, kernel_size, num_processes
):
    """Главная функция: разбивает видео по кадрам между процессами"""
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
                    i,
                )
            )

        for f in futures:
            f.result()

    elapsed = time.perf_counter() - start_time
    print(f"\nВремя обработки ({num_processes} процессов): {elapsed:.2f} c")

    concat_videos(temp_files, output_path, fps)

    for f in temp_files:
        Path(f).unlink(missing_ok=True)


# ======================= Benchmark ==========================


def run_once(video_path, num_processes, run_index):
    name = Path(video_path).stem
    out_path = f"output/{name}_p{num_processes}_r{run_index}.mp4"

    t0 = time.perf_counter()
    process_video_parallel(
        input_path=video_path,
        output_path=out_path,
        lower_rgb=RGB_RANGE[0],
        upper_rgb=RGB_RANGE[1],
        min_area=MIN_AREA,
        kernel_size=9,
        num_processes=num_processes,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, out_path


def benchmark():
    # === Очистка старых результатов ===
    output_dir = Path("output")
    if output_dir.exists():
        # удаляем все mp4 файлы из папки output
        for file in output_dir.glob("*.mp4"):
            try:
                file.unlink()
            except Exception as e:
                print(f"Не удалось удалить {file}: {e}")
    else:
        output_dir.mkdir()

    # удаляем старый CSV, если он существует
    results_csv = Path("results.csv")
    if results_csv.exists():
        try:
            results_csv.unlink()
            print("Старый results.csv удалён")
        except Exception as e:
            print(f"Не удалось удалить results.csv: {e}")

    print("Папка output очищена, начинаем новый бенчмарк...\n")

    results = []

    for workers in [1, 2, 4, 6, 8, 10]:
        print(f"\n=== Тестируем {workers} процессов ===")
        best_time = float("inf")
        best_file = None

        for run in range(5):
            print(f"\n--- Запуск {run + 1}/5 ---")

            for video_path, duration in VIDEOS:
                elapsed, out_path = run_once(video_path, workers, run)
                results.append(
                    {
                        "video": Path(video_path).name,
                        "duration": duration,
                        "processes": workers,
                        "run": run + 1,
                        "elapsed_time": elapsed,
                    }
                )
                print(f"{Path(video_path).name}: {elapsed:.2f} с")

                # сохраняем только лучший результат
                if elapsed < best_time:
                    if best_file and Path(best_file).exists():
                        Path(best_file).unlink()
                    best_time = elapsed
                    best_file = out_path
                else:
                    if Path(out_path).exists():
                        Path(out_path).unlink()

        print(
            f"Лучший результат для {workers} процессов: {best_time:.2f} с ({best_file})"
        )

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("\nВсе результаты сохранены в results.csv")


if __name__ == "__main__":
    benchmark()
