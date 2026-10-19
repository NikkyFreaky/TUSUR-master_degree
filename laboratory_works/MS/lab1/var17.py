import random
import math
import statistics


# Исходные данные
MEAN_FIRES = 4.0

P_ONE_TRUCK = 0.7
P_TWO_TRUCKS = 0.3

MEAN_ONE = 2.5
SIGMA_ONE = 0.5

MEAN_TWO = 3.5
SIGMA_TWO = 1.0


def poisson(mean):
    """Число событий по закону Пуассона."""
    limit = math.exp(-mean)
    product = 1.0
    count = 0

    while product > limit:
        product *= random.random()
        count += 1

    return count - 1


def normal(mean, sigma):
    """Нормальное время. Отрицательные значения отбрасываются."""
    while True:
        u1 = 1.0 - random.random()
        u2 = random.random()

        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(
            2.0 * math.pi * u2
        )
        value = mean + sigma * z

        if value > 0.0:
            return value


def simulate_one():
    """Одни сутки работы пожарных машин.

    Машины всегда доступны, поэтому каждая занята всё время
    тушения. Две машины дают вдвое больше машино-часов.
    """
    fires = poisson(MEAN_FIRES)

    one_truck_hours = 0.0
    two_truck_hours = 0.0

    for _ in range(fires):
        u = random.random()

        if u < P_ONE_TRUCK:
            one_truck_hours += normal(MEAN_ONE, SIGMA_ONE)
        else:
            two_truck_hours += 2.0 * normal(MEAN_TWO, SIGMA_TWO)

    total_hours = one_truck_hours + two_truck_hours

    return total_hours, fires, one_truck_hours, two_truck_hours


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
    """Многократное моделирование суток."""
    results = [[] for _ in range(4)]

    for _ in range(n):
        result = simulate_one()

        for i, value in enumerate(result):
            results[i].append(value)

    return [calculate_stats(values) for values in results]


def theoretical_means():
    """Математические ожидания машино-часов и числа пожаров."""
    one = MEAN_FIRES * P_ONE_TRUCK * MEAN_ONE
    two = MEAN_FIRES * P_TWO_TRUCKS * 2.0 * MEAN_TWO

    return one + two, MEAN_FIRES, one, two


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
    print("Моделирование занятости пожарных машин")
    print(
        f"Число пожаров за сутки: Пуассон, "
        f"среднее {MEAN_FIRES:.0f}"
    )
    print(
        f"Одна машина ({P_ONE_TRUCK:.0%}): "
        f"нормальный закон, m={MEAN_ONE}, сигма={SIGMA_ONE} ч"
    )
    print(
        f"Две машины ({P_TWO_TRUCKS:.0%}): "
        f"нормальный закон, m={MEAN_TWO}, сигма={SIGMA_TWO} ч"
    )
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    results = monte_carlo(n)
    expected_total, expected_fires, expected_one, expected_two = (
        theoretical_means()
    )

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    names = [
        ("Машино-часы за сутки", "ч"),
        ("Число пожаров", "шт."),
        ("Часы одной машины", "ч"),
        ("Часы двух машин", "ч")
    ]

    for (name, unit), stats in zip(names, results):
        print_result(name, stats, unit)

    print()
    print("ТЕОРЕТИЧЕСКИЕ СРЕДНИЕ")
    print(f"Машино-часы за сутки: {expected_total:.4f} ч")
    print(f"Число пожаров: {expected_fires:.4f} шт.")
    print(f"Часы одной машины: {expected_one:.4f} ч")
    print(f"Часы двух машин: {expected_two:.4f} ч")


if __name__ == "__main__":
    main()
