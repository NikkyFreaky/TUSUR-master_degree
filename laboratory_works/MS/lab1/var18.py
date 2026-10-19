import random
import math
import statistics


# Исходные данные
SHIFT = 8.0 * 60.0

INTERVAL_MIN = 1.0
INTERVAL_MAX = 5.0

ACTIONS = ("U1", "U2", "U3", "U4", "U5")
PROBS = (0.15, 0.20, 0.22, 0.18, 0.25)
LOSSES = (200.0, 280.0, 290.0, 220.0, 300.0)


def uniform(left, right):
    """Равномерное время на отрезке [left, right]."""
    u = random.random()
    return left + (right - left) * u


def choose_action():
    """Номер управляющего воздействия."""
    u = random.random()
    cumulative = 0.0

    for index, prob in enumerate(PROBS):
        cumulative += prob
        if u < cumulative:
            return index

    return len(PROBS) - 1


def simulate_one():
    """Одна смена: сообщения и потери от задержки решения."""
    current_time = 0.0
    total_loss = 0.0
    messages = 0
    action_losses = [0.0] * len(ACTIONS)

    while True:
        current_time += uniform(INTERVAL_MIN, INTERVAL_MAX)

        if current_time > SHIFT:
            break

        action = choose_action()
        loss = LOSSES[action]

        total_loss += loss
        action_losses[action] += loss
        messages += 1

    return (total_loss, messages, *action_losses)


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
    width = 2 + len(ACTIONS)
    results = [[] for _ in range(width)]

    for _ in range(n):
        result = simulate_one()

        for i, value in enumerate(result):
            results[i].append(value)

    return [calculate_stats(values) for values in results]


def mean_loss_per_message():
    """Средние потери на одно сообщение."""
    return sum(
        prob * loss for prob, loss in zip(PROBS, LOSSES)
    )


def expected_messages():
    """Асимптотика среднего числа сообщений за смену.

    Для процесса восстановления
    E[N(t)] ≈ t/μ + (σ²/μ² - 1)/2.
    """
    mean_interval = (INTERVAL_MIN + INTERVAL_MAX) / 2.0
    variance = (INTERVAL_MAX - INTERVAL_MIN) ** 2 / 12.0
    correction = (variance / mean_interval ** 2 - 1.0) / 2.0

    return SHIFT / mean_interval + correction


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
    print("Моделирование потерь системы управления")
    print(f"Длительность работы: {SHIFT:.0f} мин")
    print(
        f"Интервал между сообщениями: равномерно "
        f"[{INTERVAL_MIN:.0f}; {INTERVAL_MAX:.0f}] мин"
    )
    print("Управляющие воздействия и потери:")

    for name, prob, loss in zip(ACTIONS, PROBS, LOSSES):
        print(f"  {name}: p={prob:.2f}, потери {loss:.0f} руб.")

    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    results = monte_carlo(n)
    loss_mean = mean_loss_per_message()
    message_mean = expected_messages()

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    print_result("Потери за 8 часов", results[0], "руб.")
    print_result("Число сообщений", results[1], "шт.")

    print()
    print("ПОТЕРИ ПО ВОЗДЕЙСТВИЯМ")

    for name, stats in zip(ACTIONS, results[2:]):
        print_result(name, stats, "руб.")

    print()
    print("ТЕОРЕТИЧЕСКИЕ СРЕДНИЕ")
    print(f"Потери на одно сообщение: {loss_mean:.4f} руб.")
    print(f"Число сообщений: {message_mean:.4f} шт.")
    print(
        f"Потери за 8 часов: {message_mean * loss_mean:.4f} руб."
    )


if __name__ == "__main__":
    main()
