import random
import math
import statistics


# Исходные данные
MEAN_A_NORMAL = 30.0
MEAN_A_DEGRADED = 15.0
MEAN_B = 60.0

# Установленный узел A и три запасных
SPARES = 3

# Замена A: «3+1» минуты, равномерно на [2, 4]
REPLACEMENT_MIN = 2.0
REPLACEMENT_MAX = 4.0


def exponential(mean):
    """Экспоненциальное время с заданным средним."""
    u = random.random()
    return -mean * math.log(1.0 - u)


def replacement_time():
    """Время замены узла A."""
    u = random.random()
    return REPLACEMENT_MIN + (REPLACEMENT_MAX - REPLACEMENT_MIN) * u


def simulate_one():
    """Работа устройства до окончательного отказа узла A.

    Узел B стареет только во время работы устройства.
    После отказа B среднее время жизни A становится 15 минут.
    Из-за отсутствия памяти оставшееся время A генерируется заново.
    """
    spares = SPARES
    b_working = True
    b_remaining = exponential(MEAN_B)

    working_time = 0.0
    downtime = 0.0

    while True:
        mean_a = MEAN_A_NORMAL if b_working else MEAN_A_DEGRADED
        a_remaining = exponential(mean_a)

        if b_working and b_remaining < a_remaining:
            working_time += b_remaining
            b_working = False
            continue

        working_time += a_remaining

        if b_working:
            b_remaining -= a_remaining

        # Запасных узлов A больше нет
        if spares == 0:
            break

        spares -= 1
        downtime += replacement_time()

    return working_time, downtime, working_time + downtime


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
    """Многократное моделирование до окончательного отказа."""
    results = [[] for _ in range(3)]

    for _ in range(n):
        result = simulate_one()

        for i, value in enumerate(result):
            results[i].append(value)

    return [calculate_stats(values) for values in results]


def theoretical_working_time():
    """Математическое ожидание времени работы, минуты.

    Состояние (k, B): ещё k отказов A, узел B исправен или нет.
    Пока B исправен, A и B соревнуются как два экспоненциальных потока.
    """
    rate_a = 1.0 / MEAN_A_NORMAL
    rate_b = 1.0 / MEAN_B
    rate = rate_a + rate_b

    expected_up = 0.0

    for failures_left in range(1, SPARES + 2):
        degraded = failures_left * MEAN_A_DEGRADED
        expected_up = (
            1.0 / rate
            + (rate_a / rate) * expected_up
            + (rate_b / rate) * degraded
        )

    return expected_up


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
    print("Моделирование надёжности устройства")
    print("Узлы: A необходим для работы, B поддерживает режим A")
    print(
        f"Среднее время A: {MEAN_A_NORMAL:.0f} мин, "
        f"после отказа B: {MEAN_A_DEGRADED:.0f} мин"
    )
    print(f"Среднее время B: {MEAN_B:.0f} мин")
    print(f"Запасных узлов A: {SPARES}")
    print(
        f"Замена A: равномерно "
        f"[{REPLACEMENT_MIN:.0f}; {REPLACEMENT_MAX:.0f}] мин"
    )
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    results = monte_carlo(n)
    expected = theoretical_working_time()
    mean_replacement = (REPLACEMENT_MIN + REPLACEMENT_MAX) / 2.0

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    names = [
        ("Время в работающем состоянии", "мин"),
        ("Время замен узла A", "мин"),
        ("Время до окончательного отказа", "мин")
    ]

    for (name, unit), stats in zip(names, results):
        print_result(name, stats, unit)

    print()
    print("ТЕОРЕТИЧЕСКОЕ СРЕДНЕЕ")
    print(f"Время в работающем состоянии: {expected:.4f} мин")
    print(
        f"Время замен узла A: "
        f"{SPARES * mean_replacement:.4f} мин"
    )


if __name__ == "__main__":
    main()
