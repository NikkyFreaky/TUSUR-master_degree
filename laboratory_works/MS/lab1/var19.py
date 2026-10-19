import random
import math
import statistics


# Исходные данные
# Служба работает с 16 до 24 часов
SHIFT = 8.0 * 60.0

# Интервал «20+5» минут: равномерно на [15, 25]
INTERVAL_MIN = 15.0
INTERVAL_MAX = 25.0

P_SHORT = 0.4
P_LONG = 0.6
MEAN_SHORT = 15.0
MEAN_LONG = 25.0


def uniform(left, right):
    """Равномерное время на отрезке [left, right]."""
    u = random.random()
    return left + (right - left) * u


def exponential(mean):
    """Экспоненциальное время с заданным средним."""
    u = random.random()
    return -mean * math.log(1.0 - u)


def service_time():
    """Длительность разговора: 15 мин в 40% случаев, иначе 25."""
    if random.random() < P_SHORT:
        return exponential(MEAN_SHORT)

    return exponential(MEAN_LONG)


def simulate_one():
    """Одна смена. Занятый номер клиент не ждёт, а кладёт трубку."""
    current_time = 0.0
    free_time = 0.0

    served = 0
    lost = 0

    while True:
        current_time += uniform(INTERVAL_MIN, INTERVAL_MAX)

        if current_time > SHIFT:
            break

        if current_time >= free_time:
            served += 1
            free_time = current_time + service_time()
        else:
            lost += 1

    return served, lost


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
    served_counts = []
    lost_counts = []
    percentages = []

    for _ in range(n):
        served, lost = simulate_one()
        total = served + lost

        served_counts.append(served)
        lost_counts.append(lost)
        percentages.append(100.0 * served / total)

    return (
        calculate_stats(percentages),
        calculate_stats(served_counts),
        calculate_stats(lost_counts)
    )


def steady_state_percentage():
    """Доля обслуженных в длинной смене, проценты.

    После принятого звонка теряется N(S) следующих:
    их интервалы укладываются в длительность разговора S.
    """
    step = 0.05
    limit = 500.0
    count = int(limit / step)
    renewal = [0.0] * (count + 1)

    for k in range(1, count + 1):
        t = k * step

        if t < INTERVAL_MIN:
            continue

        if t < INTERVAL_MAX:
            distribution = (t - INTERVAL_MIN) / (
                INTERVAL_MAX - INTERVAL_MIN
            )
        else:
            distribution = 1.0

        left = int(INTERVAL_MIN / step)
        right = int(min(INTERVAL_MAX, t) / step)
        density = 1.0 / (INTERVAL_MAX - INTERVAL_MIN)
        integral = 0.0

        for j in range(left, right):
            integral += renewal[k - j]

        renewal[k] = distribution + integral * density * step

    expected_lost = 0.0

    for k in range(count + 1):
        t = k * step
        density = (
            P_SHORT / MEAN_SHORT * math.exp(-t / MEAN_SHORT)
            + P_LONG / MEAN_LONG * math.exp(-t / MEAN_LONG)
        )
        expected_lost += renewal[k] * density * step

    served_fraction = 1.0 / (1.0 + expected_lost)
    return 100.0 * served_fraction


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
    print("Моделирование службы «Телефон доверия»")
    print("Время работы: с 16 до 24 часов")
    print(
        f"Интервал между звонками: равномерно "
        f"[{INTERVAL_MIN:.0f}; {INTERVAL_MAX:.0f}] мин"
    )
    print(
        f"Разговор: экспоненциальный закон, "
        f"{P_SHORT:.0%} случаев по {MEAN_SHORT:.0f} мин, "
        f"{P_LONG:.0%} по {MEAN_LONG:.0f} мин"
    )
    print("Занятый номер: клиент кладёт трубку")
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    percent_stats, served_stats, lost_stats = monte_carlo(n)
    steady = steady_state_percentage()

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    print_result("Доля обслуженных клиентов", percent_stats, "%")
    print_result("Обслуженные клиенты", served_stats, "чел.")
    print_result("Клиенты, получившие «занято»", lost_stats, "чел.")

    served_mean = served_stats[0]
    lost_mean = lost_stats[0]
    total_mean = served_mean + lost_mean

    print()
    print(
        f"Процент по средним числам клиентов: "
        f"{100.0 * served_mean / total_mean:.4f}%"
    )

    print()
    print("СТАЦИОНАРНАЯ ДОЛЯ")
    print(
        "Для очень длинной смены, когда начало в 16:00 "
        f"уже не влияет: {steady:.4f}%"
    )


if __name__ == "__main__":
    main()
