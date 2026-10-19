import random
import math
import statistics


# Исходные данные
SHIFT_HOURS = 8.0

# f(t) = a * exp(-a * t), a = 3 партии в час
ARRIVAL_RATE = 3.0
MEAN_INTERVAL = 1.0 / ARRIVAL_RATE

BATCH_SIZES = (1000, 1200, 1400)
BATCH_PROBS = (0.3, 0.6, 0.1)

BRIGADE_PROBS = (0.3, 0.5, 0.2)


def exponential(mean):
    """Экспоненциальное время с заданным средним."""
    u = random.random()
    return -mean * math.log(1.0 - u)


def pick(values, probs):
    """Выбор значения по заданным вероятностям."""
    u = random.random()
    cumulative = 0.0

    for value, prob in zip(values, probs):
        cumulative += prob
        if u < cumulative:
            return value

    return values[-1]


def simulate_one():
    """Одна смена: партии приходят и распределяются по бригадам."""
    parts = [0, 0, 0]
    batches = 0
    current_time = 0.0

    while True:
        current_time += exponential(MEAN_INTERVAL)

        if current_time > SHIFT_HOURS:
            break

        size = pick(BATCH_SIZES, BATCH_PROBS)
        brigade = pick((0, 1, 2), BRIGADE_PROBS)

        parts[brigade] += size
        batches += 1

    return parts[0], parts[1], parts[2], batches


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
    results = [[] for _ in range(4)]

    for _ in range(n):
        result = simulate_one()

        for i, value in enumerate(result):
            results[i].append(value)

    return [calculate_stats(values) for values in results]


def theoretical_means():
    """Математические ожидания числа деталей и партий."""
    mean_size = sum(
        size * prob
        for size, prob in zip(BATCH_SIZES, BATCH_PROBS)
    )
    mean_batches = ARRIVAL_RATE * SHIFT_HOURS
    mean_parts = [
        mean_batches * mean_size * prob
        for prob in BRIGADE_PROBS
    ]

    return mean_parts, mean_batches


def print_result(name, stats, unit):
    """Вывод среднего значения и доверительного интервала."""
    mean, lower, upper = stats

    print(f"{name}: {mean:.4f} {unit}")

    if lower is not None:
        print(
            f"  95%-й доверительный интервал: "
            f"[{lower:.4f}; {upper:.4f}]"
        )


def main():
    print("Моделирование поставок комплектующих")
    print(f"Длительность смены: {SHIFT_HOURS:.0f} ч")
    print(
        f"Интервалы между партиями: экспоненциальный закон, "
        f"a = {ARRIVAL_RATE:.0f} партии в час"
    )
    print(
        "Размеры партий: "
        + ", ".join(
            f"{size} (p={prob})"
            for size, prob in zip(BATCH_SIZES, BATCH_PROBS)
        )
    )
    print(
        "Вероятности бригад: "
        + ", ".join(
            f"P({i + 1})={prob}"
            for i, prob in enumerate(BRIGADE_PROBS)
        )
    )
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    results = monte_carlo(n)
    expected_parts, expected_batches = theoretical_means()

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    names = [
        ("Бригада 1", "шт."),
        ("Бригада 2", "шт."),
        ("Бригада 3", "шт."),
        ("Число партий", "шт.")
    ]

    for (name, unit), stats in zip(names, results):
        print_result(name, stats, unit)

    total = sum(stats[0] for stats in results[:3])

    print()
    print(f"Всего деталей за смену: {total:.4f} шт.")

    print()
    print("ТЕОРЕТИЧЕСКИЕ СРЕДНИЕ")

    for i, value in enumerate(expected_parts, start=1):
        print(f"Бригада {i}: {value:.4f} шт.")

    print(f"Число партий: {expected_batches:.4f} шт.")
    print(
        f"Всего деталей: {sum(expected_parts):.4f} шт."
    )


if __name__ == "__main__":
    main()
