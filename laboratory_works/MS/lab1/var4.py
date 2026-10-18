
import random
import math
import statistics


# Исходные данные
DEFECT_PROBABILITY = 0.1
ARRIVAL_RATE = 2.0

FIRST_THRESHOLD = 20
SECOND_THRESHOLD = 10

CONTROL_INTERVAL = 3


def exponential(rate, rng):
    """Интервал поступления деталей (экспоненциальный)."""
    u = rng.random()
    return -math.log1p(-u) / rate


def is_defective(rng):
    """Определение брака с вероятностью 0.1."""
    return rng.random() < DEFECT_PROBABILITY


def simulate_one(rng):
    """Одна реализация работы станка до остановки."""
    time = 0.0
    total_parts = 0

    defective_count = 0
    stage = 1

    while True:
        # Поступление очередной детали
        time += exponential(ARRIVAL_RATE, rng)
        total_parts += 1

        if stage == 1:
            # Проверяем только каждую третью деталь
            if total_parts % CONTROL_INTERVAL == 0:

                if is_defective(rng):
                    defective_count += 1

                    if defective_count == FIRST_THRESHOLD:
                        # Переход к сплошному контролю
                        stage = 2
                        defective_count = 0

        else:
            # На втором этапе проверяем каждую деталь
            if is_defective(rng):
                defective_count += 1

                if defective_count == SECOND_THRESHOLD:
                    # Станок останавливается
                    return time, total_parts


def calculate_stats(values):
    """Среднее значение и 95%-й доверительный интервал."""
    n = len(values)
    mean = statistics.mean(values)

    if n == 1:
        return mean, None, None

    std = statistics.stdev(values)
    se = std / math.sqrt(n)

    lower = mean - 1.96 * se
    upper = mean + 1.96 * se

    return mean, lower, upper


def monte_carlo(n, seed=None):
    """Многократное моделирование работы станка."""
    rng = random.Random(seed)

    times = []
    parts = []

    for _ in range(n):
        t, count = simulate_one(rng)
        times.append(t)
        parts.append(count)

    time_stats = calculate_stats(times)
    parts_stats = calculate_stats(parts)

    return time_stats, parts_stats


def main():
    print("Моделирование работы токарного станка")
    print(f"Вероятность брака: {DEFECT_PROBABILITY}")
    print(f"Интенсивность: {ARRIVAL_RATE} детали/мин")
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    time_stats, parts_stats = monte_carlo(n)

    avg_time, lower, upper = time_stats
    avg_parts = parts_stats[0]

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    print(f"Среднее время до остановки: {avg_time:.4f} мин")
    print(f"В часах: {avg_time / 60:.4f} ч")
    print(f"Среднее количество деталей: {avg_parts:.2f}")

    if lower is not None:
        print(
            f"95%-й доверительный интервал: "
            f"[{lower:.4f}; {upper:.4f}] мин"
        )


if __name__ == "__main__":
    main()
