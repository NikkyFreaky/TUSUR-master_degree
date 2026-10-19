import random
import math
import statistics


# Исходные данные
SHIFT = 8.0 * 60.0

MEAN_TRANSPORT = 5.0
PROCESS_MIN = 2.0
PROCESS_MAX = 6.0
SCRAP_LIMIT = 14.0


def exponential(mean):
    """Экспоненциальное время с заданным средним."""
    u = random.random()
    return -mean * math.log(1.0 - u)


def process_time():
    """Равномерное время обработки на отрезке [2, 6]."""
    u = random.random()
    return PROCESS_MIN + (PROCESS_MAX - PROCESS_MIN) * u


def simulate_one():
    """Одна смена: детали обрабатываются одна за другой.

    Деталь учитывается, только если транспортировка и обработка
    завершились до конца смены. Брак — если их сумма больше 14 минут.
    """
    current_time = 0.0
    good = 0
    bad = 0

    while True:
        transport = exponential(MEAN_TRANSPORT)
        processing = process_time()
        cycle = transport + processing

        if current_time + cycle > SHIFT:
            break

        current_time += cycle

        if cycle > SCRAP_LIMIT:
            bad += 1
        else:
            good += 1

    return good, bad


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


def monte_carlo(n):
    """Многократное моделирование смены."""
    good_counts = []
    bad_counts = []

    for _ in range(n):
        good, bad = simulate_one()
        good_counts.append(good)
        bad_counts.append(bad)

    return (
        calculate_stats(good_counts),
        calculate_stats(bad_counts)
    )


def defect_probability():
    """Точная вероятность брака: P(T + U > 14)."""
    return (MEAN_TRANSPORT / (PROCESS_MAX - PROCESS_MIN)) * (
        math.exp(-(SCRAP_LIMIT - PROCESS_MAX) / MEAN_TRANSPORT)
        - math.exp(-(SCRAP_LIMIT - PROCESS_MIN) / MEAN_TRANSPORT)
    )


def print_result(name, stats):
    """Вывод среднего значения и доверительного интервала."""
    mean, lower, upper = stats

    print(f"{name}: {mean:.4f} шт.")

    if lower is not None:
        print(
            f"  95%-й доверительный интервал: "
            f"[{lower:.4f}; {upper:.4f}]"
        )


def main():
    print("Моделирование производства деталей")
    print(f"Длительность смены: {SHIFT:.0f} мин")
    print(
        f"Транспортировка: экспоненциальный закон, "
        f"среднее {MEAN_TRANSPORT:.0f} мин"
    )
    print(
        f"Обработка: равномерно "
        f"[{PROCESS_MIN:.0f}; {PROCESS_MAX:.0f}] мин"
    )
    print(f"Брак, если сумма времён больше {SCRAP_LIMIT:.0f} мин")
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    good_stats, bad_stats = monte_carlo(n)
    p_bad = defect_probability()
    mean_cycle = MEAN_TRANSPORT + (PROCESS_MIN + PROCESS_MAX) / 2.0
    mean_count = SHIFT / mean_cycle

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    print_result("Годные детали", good_stats)
    print_result("Бракованные детали", bad_stats)

    good_mean = good_stats[0]
    bad_mean = bad_stats[0]
    total_mean = good_mean + bad_mean

    print()
    print("ДОПОЛНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ")
    print(f"Среднее число деталей за смену: {total_mean:.4f} шт.")

    if total_mean > 0:
        print(
            f"Доля брака: {bad_mean / total_mean * 100:.2f}%"
        )

    print()
    print("ТЕОРЕТИЧЕСКИЕ ХАРАКТЕРИСТИКИ")
    print(f"Вероятность брака: {p_bad:.6f}")
    print(f"Вероятность годной детали: {1.0 - p_bad:.6f}")
    print(f"Среднее время цикла: {mean_cycle:.4f} мин")
    print(
        "Оценка без учёта края смены: "
        f"годных {mean_count * (1.0 - p_bad):.4f}, "
        f"брака {mean_count * p_bad:.4f}"
    )


if __name__ == "__main__":
    main()
