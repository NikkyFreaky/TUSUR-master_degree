import random
import math
import statistics


# Исходные данные
BLOCKS = 1000

# P(0), P(1), P(2 и более)
P_NONE = 0.95
P_SINGLE = 0.03
P_MULTIPLE = 0.02

# Равномерное время передачи: «5+2» и «6+2» секунды.
# Среднее 5 и 6 с, отклонение 2 с, то есть интервалы [3, 7] и [4, 8].
AX_MIN = 3.0
AX_MAX = 7.0
XB_MIN = 4.0
XB_MAX = 8.0

# Коррекция одиночной ошибки: экспоненциальный закон, среднее 1 с
MEAN_CORRECTION = 1.0


def uniform(left, right):
    """Равномерное время на отрезке [left, right]."""
    u = random.random()
    return left + (right - left) * u


def exponential(mean):
    """Экспоненциальное время с заданным средним."""
    u = random.random()
    return -mean * math.log(1.0 - u)


def generate_error():
    """Тип ошибки при одной передаче блока."""
    u = random.random()

    if u < P_NONE:
        return 0

    if u < P_NONE + P_SINGLE:
        return 1

    return 2


def transmit_hop(t_min, t_max):
    """Передача блока на один участок до успешного приёма.

    Ноль ошибок: участок пройден.
    Одна ошибка: коррекция на приёмном конце, участок пройден.
    Две и более: повтор передачи с предыдущего пункта.
    """
    transmission_time = 0.0
    correction_time = 0.0
    retransmissions = 0
    corrections = 0

    first_attempt = True

    while True:
        transmission_time += uniform(t_min, t_max)

        if not first_attempt:
            retransmissions += 1

        first_attempt = False
        error = generate_error()

        if error == 0:
            break

        if error == 1:
            correction_time += exponential(MEAN_CORRECTION)
            corrections += 1
            break

    return (
        transmission_time,
        correction_time,
        retransmissions,
        corrections
    )


def simulate_one():
    """Передача 1000 блоков по маршруту A -> X -> B."""
    ax_time = 0.0
    xb_time = 0.0
    correction_time = 0.0
    retransmissions = 0
    corrections = 0

    for _ in range(BLOCKS):
        ax_tx, ax_corr, ax_re, ax_c = transmit_hop(AX_MIN, AX_MAX)
        xb_tx, xb_corr, xb_re, xb_c = transmit_hop(XB_MIN, XB_MAX)

        ax_time += ax_tx
        xb_time += xb_tx
        correction_time += ax_corr + xb_corr
        retransmissions += ax_re + xb_re
        corrections += ax_c + xb_c

    total_time = ax_time + xb_time + correction_time

    return (
        total_time,
        ax_time,
        xb_time,
        correction_time,
        retransmissions,
        corrections
    )


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
    """Многократное моделирование передачи 1000 блоков."""
    results = [[] for _ in range(6)]

    for _ in range(n):
        result = simulate_one()

        for i, value in enumerate(result):
            results[i].append(value)

    return [calculate_stats(values) for values in results]


def theoretical_total_time():
    """Математическое ожидание суммарного времени, секунды.

    На каждом участке число попыток геометрическое
    с вероятностью успеха P(0) + P(1). Коррекция выполняется
    только на успешной попытке с одиночной ошибкой.
    """
    success = P_NONE + P_SINGLE
    mean_ax = (AX_MIN + AX_MAX) / 2.0
    mean_xb = (XB_MIN + XB_MAX) / 2.0

    per_block = (
        (mean_ax + mean_xb) / success
        + 2.0 * P_SINGLE / success * MEAN_CORRECTION
    )

    return BLOCKS * per_block


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
    print("Моделирование системы передачи данных A -> X -> B")
    print(f"Число блоков: {BLOCKS}")
    print(
        f"Вероятности ошибок: "
        f"P(0)={P_NONE}, P(1)={P_SINGLE}, "
        f"P(2 и более)={P_MULTIPLE}"
    )
    print(
        f"Передача A -> X: равномерно "
        f"[{AX_MIN:.0f}; {AX_MAX:.0f}] с"
    )
    print(
        f"Передача X -> B: равномерно "
        f"[{XB_MIN:.0f}; {XB_MAX:.0f}] с"
    )
    print(
        f"Коррекция одиночной ошибки: "
        f"экспоненциальный закон, среднее {MEAN_CORRECTION:.0f} с"
    )
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    results = monte_carlo(n)
    expected = theoretical_total_time()

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print(f"Блоков в каждой реализации: {BLOCKS}")
    print()

    names = [
        ("Суммарные затраты времени", "с"),
        ("Время передачи A -> X", "с"),
        ("Время передачи X -> B", "с"),
        ("Время коррекции ошибок", "с"),
        ("Число повторных передач", "шт."),
        ("Число коррекций", "шт.")
    ]

    for (name, unit), stats in zip(names, results):
        print_result(name, stats, unit)

    print()
    print("ТЕОРЕТИЧЕСКОЕ СРЕДНЕЕ")
    print(f"Суммарные затраты времени: {expected:.4f} с")
    print(f"В часах: {expected / 3600.0:.4f} ч")


if __name__ == "__main__":
    main()
