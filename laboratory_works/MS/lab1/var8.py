
import random
import math


# Исходные данные
SIMULATION_TIME = 600.0
MEAN_TRANSITION_TIME = 2.0

# Матрица переходных вероятностей
P = [
    [0.5, 0.3, 0.2],
    [0.5, 0.0, 0.5],
    [0.3, 0.25, 0.45]
]

INITIAL_STATE = 0  # S1


def exponential(mean):
    """Генерация экспоненциального времени."""
    u = random.random()
    return -mean * math.log(1 - u)


def next_state(state):
    """Выбор следующего состояния по матрице P."""
    u = random.random()

    cumulative_probability = 0.0

    for i, probability in enumerate(P[state]):
        cumulative_probability += probability

        if u < cumulative_probability:
            return i

    return len(P[state]) - 1


def simulate_one():
    """Одна реализация работы системы за 600 секунд."""
    current_time = 0.0
    state = INITIAL_STATE

    transitions = 0

    while True:
        # Время до следующего перехода
        dt = exponential(MEAN_TRANSITION_TIME)

        # Проверяем, произойдёт ли переход до 600 секунд
        if current_time + dt > SIMULATION_TIME:
            break

        # Увеличиваем время
        current_time += dt

        # Определяем новое состояние
        state = next_state(state)

        transitions += 1

    return state, transitions


def confidence_interval(p, n):
    """Приближённый 95%-й доверительный интервал."""
    se = math.sqrt(p * (1 - p) / n)

    lower = max(0.0, p - 1.96 * se)
    upper = min(1.0, p + 1.96 * se)

    return lower, upper


def monte_carlo(n):
    """Моделирование методом Монте-Карло."""
    state_counts = [0, 0, 0]
    total_transitions = 0

    for _ in range(n):
        final_state, transitions = simulate_one()

        state_counts[final_state] += 1
        total_transitions += transitions

    probabilities = [count / n for count in state_counts]

    intervals = [
        confidence_interval(p, n)
        for p in probabilities
    ]

    avg_transitions = total_transitions / n

    return probabilities, intervals, avg_transitions


def main():
    print("Моделирование однородной цепи Маркова")
    print(f"Начальное состояние: S{INITIAL_STATE + 1}")
    print(f"Время моделирования: {SIMULATION_TIME} секунд")
    print(
        f"Среднее время между переходами: "
        f"{MEAN_TRANSITION_TIME} секунды"
    )
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    probabilities, intervals, avg_transitions = monte_carlo(n)

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    for i in range(3):
        p = probabilities[i]
        lower, upper = intervals[i]

        print(f"Состояние S{i + 1}:")
        print(f"  Вероятность: {p:.6f}")
        print(f"  В процентах: {p * 100:.2f}%")
        print(
            f"  95%-й доверительный интервал: "
            f"[{lower:.6f}; {upper:.6f}]"
        )
        print()

    print("ДОПОЛНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ")
    print(f"Среднее количество переходов: {avg_transitions:.4f}")
    print(
        f"Теоретическое среднее количество: "
        f"{SIMULATION_TIME / MEAN_TRANSITION_TIME:.4f}"
    )

    print(f"Сумма вероятностей: {sum(probabilities):.6f}")


if __name__ == "__main__":
    main()
