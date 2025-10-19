import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

COLOR_RANGES = {
    "orange": ((214, 117, 13), (255, 197, 93)),
    "red": ((150, 20, 30), (230, 80, 90)),
    "blue": ((25, 80, 160), (90, 160, 255)),
}

DEFAULT_RGB_RANGE = ((0, 0, 0), (255, 255, 255))

MIN_AREA = 250
KERNEL_SIZE = 9
NUM_RUNS = 5
PROCESS_COUNTS = [1, 2, 4, 6, 8, 10]


def get_color_range(group_prefix):
    return COLOR_RANGES.get(group_prefix, DEFAULT_RGB_RANGE)


def get_video_prefix(filename):
    parts = filename.split("_")
    for i, part in enumerate(parts):
        if part.endswith("p") and any(c.isdigit() for c in part):
            return "_".join(parts[:i])
    return filename.split("_")[0]


def scan_input_videos(input_dir="input"):
    input_path = Path(input_dir)
    if not input_path.exists():
        input_path.mkdir()
        print(f"Создана папка {input_dir}. Поместите туда видеофайлы.")
        return {}

    video_groups = defaultdict(list)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    for video_file in input_path.iterdir():
        if video_file.suffix.lower() in video_extensions:
            prefix = get_video_prefix(video_file.stem)

            cap = cv.VideoCapture(str(video_file))
            fps = cap.get(cv.CAP_PROP_FPS)
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            duration = int(frame_count / fps) if fps > 0 else 0
            cap.release()

            video_groups[prefix].append(
                {"path": str(video_file), "duration": duration, "name": video_file.name}
            )

    return video_groups


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
    cap = cv.VideoCapture(input_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

    morph_kernel = np.ones((3, 3), np.uint8)
    dilate_kernel = np.ones((3, 3), np.uint8)

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
    thickness = 2 + part_index % 3

    lower_rgb_array = np.array(lower_rgb, dtype=np.uint8)
    upper_rgb_array = np.array(upper_rgb, dtype=np.uint8)

    frames_to_process = end_frame - start_frame

    for _ in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mask = cv.inRange(rgb, lower_rgb_array, upper_rgb_array)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, morph_kernel)

        n_labels, labels, stats, _ = cv.connectedComponentsWithStats(
            mask, connectivity=8
        )

        region_mask = np.zeros((h, w), dtype=np.uint8)

        valid_regions = []
        for i in range(1, n_labels):
            area = stats[i, cv.CC_STAT_AREA]
            if area >= min_area:
                x = stats[i, cv.CC_STAT_LEFT]
                y = stats[i, cv.CC_STAT_TOP]
                w_box = stats[i, cv.CC_STAT_WIDTH]
                h_box = stats[i, cv.CC_STAT_HEIGHT]

                cv.rectangle(region_mask, (x, y), (x + w_box, y + h_box), 255, -1)
                valid_regions.append((x, y, w_box, h_box))

        region_mask = cv.dilate(region_mask, dilate_kernel, iterations=1)

        blurred = cv.filter2D(frame, -1, kernel)

        result = blurred.copy()

        annotated = frame.copy()
        for x, y, w_box, h_box in valid_regions:
            cv.rectangle(annotated, (x, y), (x + w_box, y + h_box), color, thickness)
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

        cv.copyTo(annotated, region_mask, result)

        out.write(result)

    cap.release()
    out.release()


def concat_videos(temp_files, output_path, fps):
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
    print(f"  Сохранено в {output_path}")


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
                    i,
                )
            )

        for f in futures:
            f.result()

    elapsed = time.perf_counter() - start_time
    print(f"  Время обработки ({num_processes} процесса/ов): {elapsed:.2f} с")

    concat_videos(temp_files, output_path, fps)

    for f in temp_files:
        Path(f).unlink(missing_ok=True)


# ======================= Benchmark ==========================


def run_once(video_path, num_processes, run_index, video_prefix):
    """Запуск одного теста"""
    video_name = Path(video_path).stem
    out_path = f"output/{video_prefix}_p{num_processes}_r{run_index}_{video_name}.mp4"

    rgb_range = get_color_range(video_prefix)

    t0 = time.perf_counter()
    process_video_parallel(
        input_path=video_path,
        output_path=out_path,
        lower_rgb=rgb_range[0],
        upper_rgb=rgb_range[1],
        min_area=MIN_AREA,
        kernel_size=KERNEL_SIZE,
        num_processes=num_processes,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, out_path


def benchmark():
    """Главная функция бенчмарка"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("Сканирование папки input...")
    video_groups = scan_input_videos()

    if not video_groups:
        print("В папке input не найдено видеофайлов!")
        return

    print(f"\nНайдена/ы {len(video_groups)} группа/ы видео:")
    for prefix, videos in video_groups.items():
        print(f"  - {prefix}: {len(videos)} видео")
        for v in videos:
            print(f"    • {v['name']} ({v['duration']}s)")

    output_dir = Path("output")
    if output_dir.exists():
        for file in output_dir.glob("*.mp4"):
            try:
                file.unlink()
            except Exception as e:
                print(f"Не удалось удалить {file}: {e}")
    else:
        output_dir.mkdir()

    plots_dir = Path("plots")
    if plots_dir.exists():
        for file in plots_dir.glob("*.png"):
            try:
                file.unlink()
            except Exception as e:
                print(f"Не удалось удалить {file}: {e}")
    else:
        plots_dir.mkdir()

    results_csv = Path("results.csv")
    if results_csv.exists():
        results_csv.unlink()

    print("\nПапки output и plots очищены, начинаем бенчмарк...\n")

    all_results = []

    for prefix, videos in video_groups.items():
        print(f"\n{'=' * 60}")
        print(f"Группа: {prefix}")

        rgb_range = get_color_range(prefix)
        print(f"Цветовой диапазон: RGB {rgb_range[0]} - {rgb_range[1]}")

        for workers in PROCESS_COUNTS:
            print(f"\n=== Тестируем {workers} процесс/ов ===")
            best_times = {}

            for run in range(NUM_RUNS):
                print(f"\n--- Тест {run + 1}/{NUM_RUNS} ---")

                for video_info in videos:
                    video_path = video_info["path"]
                    duration = video_info["duration"]
                    video_name = video_info["name"]

                    elapsed, out_path = run_once(video_path, workers, run, prefix)

                    all_results.append(
                        {
                            "group": prefix,
                            "video": video_name,
                            "duration": duration,
                            "processes": workers,
                            "run": run + 1,
                            "elapsed_time": elapsed,
                            "rgb_lower": str(rgb_range[0]),
                            "rgb_upper": str(rgb_range[1]),
                        }
                    )

                    print(f"  {video_name} — {elapsed:.2f} с")

                    key = (video_name, workers)
                    if key not in best_times or elapsed < best_times[key][0]:
                        if key in best_times and Path(best_times[key][1]).exists():
                            Path(best_times[key][1]).unlink()
                        best_times[key] = (elapsed, out_path)
                    else:
                        if Path(out_path).exists():
                            Path(out_path).unlink()

            for video_info in videos:
                video_name = video_info["name"]
                key = (video_name, workers)
                if key in best_times:
                    print(
                        f"\nЛучший для {video_name} ({workers} процесса/ов): {best_times[key][0]:.2f} с"
                    )

    df = pd.DataFrame(all_results)
    df.to_csv(results_csv, index=False)
    print(f"\nВсе результаты сохранены в {results_csv}")

    sns.set_theme(style="whitegrid")

    for prefix in video_groups.keys():
        subset = df[df["group"] == prefix]

        if subset.empty:
            continue

        summary = subset.groupby("processes", as_index=False)["elapsed_time"].mean()

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=summary, x="processes", y="elapsed_time", marker="o", linewidth=2.5
        )
        plt.title(f"Average Processing Time — {prefix}", fontsize=14, fontweight="bold")
        plt.xlabel("Number of Processes", fontsize=12)
        plt.ylabel("Average Time (seconds)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname_avg = plots_dir / f"{prefix}_avg_time.png"
        plt.savefig(fname_avg, dpi=200)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=subset, x="processes", y="elapsed_time", palette="Set2")
        plt.title(f"Time Distribution — {prefix}", fontsize=14, fontweight="bold")
        plt.xlabel("Number of Processes", fontsize=12)
        plt.ylabel("Execution Time (seconds)", fontsize=12)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        fname_box = plots_dir / f"{prefix}_boxplot.png"
        plt.savefig(fname_box, dpi=200)
        plt.close()

        print(f"Графики для группы '{prefix}': {fname_avg.name}, {fname_box.name}")

    print(f"\nПостроение графиков завершено. Все результаты сохранены в папке 'plots/'")
    print(f"Лучшие видео сохранены в папке 'output/'")


if __name__ == "__main__":
    benchmark()
