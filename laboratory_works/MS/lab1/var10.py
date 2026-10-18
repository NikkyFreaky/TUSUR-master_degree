
import random
import math
import statistics


# Исходные данные
SIMULATION_TIME = 600.0  # 10 часов в минутах

ERROR_PROBABILITY = 0.1
SUCCESS_PROBABILITY = 0.9

MIN_TRANSMISSION_TIME = 3.0
MAX_TRANSMISSION_TIME = 7.0

DEFAULT_REALIZATIONS = 100

# Обозначения состояний
S0 = 0  # Нет сбоев
S1 = 1  # Один сбой подряд
S2 = 2  # Два сбоя подряд
S3 = 3  # Аварийный режим


def generate_transmission_time():
    """Время передачи сообщения от 3 до 7 минут."""
    u = random.random()

    return MIN_TRANSMISSION_TIME + (
        MAX_TRANSMISSION_TIME - MIN_TRANSMISSION_TIME
    ) * u


def next_state(state):
    """Определение состояния после передачи сообщения."""
    u = random.random()

    if u < ERROR_PROBABILITY:
        # Сбой: увеличиваем число последовательных ошибок
        return min(state + 1, S3)

    # Успешный приём сбрасывает счётчик сбоев
    return S0


def simulate_one():
    """Одна реализация работы системы за 10 часов."""
    current_time = 0.0
    state = S0

    # Время нахождения в каждом состоянии
    state_times = [0.0] * 4

    messages = 0
    errors = 0
    emergency_entries = 0

    while current_time < SIMULATION_TIME:
        # Генерация времени передачи
        dt = generate_transmission_time()

        # Не выходим за границу моделирования
        remaining = SIMULATION_TIME - current_time
        actual_dt = min(dt, remaining)

        # Во время передачи сохраняется предыдущее состояние
        state_times[state] += actual_dt
        current_time += actual_dt

        # Если сообщение не успело передаться,
        # новое состояние не определяется
        if dt > actual_dt:
            break

        # Сообщение передано полностью
        messages += 1

        old_state = state
        state = next_state(state)

        if state != S0:
            errors += 1

        # Новый вход в аварийный режим
        if old_state == S2 and state == S3:
            emergency_entries += 1

    return (
        state,
        state_times,
        messages,
        errors,
        emergency_entries
    )


def confidence_interval(values):
    """Среднее и 95%-й доверительный интервал."""
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
    """Моделирование методом Монте-Карло."""
    final_counts = [0] * 4
    time_fractions = [[] for _ in range(4)]

    message_counts = []
    error_counts = []
    emergency_counts = []

    for _ in range(n):
        result = simulate_one()

        final_state, times, messages, errors, emergencies = result

        final_counts[final_state] += 1

        for i in range(4):
            time_fractions[i].append(
                times[i] / SIMULATION_TIME
            )

        message_counts.append(messages)
        error_counts.append(errors)
        emergency_counts.append(emergencies)

    # Вероятности состояний в конце 10 часов
    final_probabilities = [
        count / n for count in final_counts
    ]

    # Средние доли времени в состояниях
    time_stats = [
        confidence_interval(values)
        for values in time_fractions
    ]

    return (
        final_probabilities,
        time_stats,
        confidence_interval(message_counts),
        confidence_interval(error_counts),
        confidence_interval(emergency_counts)
    )


def main():
    print("Моделирование системы передачи данных")
    print(f"Продолжительность: {SIMULATION_TIME} минут")
    print(f"Вероятность сбоя: {ERROR_PROBABILITY}")
    print(f"Вероятность правильного приёма: {SUCCESS_PROBABILITY}")
    print()

    value = input(
        f"Введите количество реализаций "
        f"[{DEFAULT_REALIZATIONS}]: "
    ).strip()

    n = int(value) if value else DEFAULT_REALIZATIONS

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    (
        probabilities,
        time_stats,
        message_stats,
        error_stats,
        emergency_stats
    ) = monte_carlo(n)

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")

    print()
    print("ВЕРОЯТНОСТИ СОСТОЯНИЙ ЧЕРЕЗ 10 ЧАСОВ")

    for i, p in enumerate(probabilities):
        print(f"S{i}: {p:.6f} ({p * 100:.2f}%)")

    normal_probability = sum(probabilities[:3])
    emergency_probability = probabilities[3]

    print()
    print(f"Нормальный режим: {normal_probability:.6f}")
    print(f"Аварийный режим: {emergency_probability:.6f}")

    print()
    print("СРЕДНИЕ ДОЛИ ВРЕМЕНИ В СОСТОЯНИЯХ")

    for i, (mean, lower, upper) in enumerate(time_stats):
        print(f"S{i}: {mean:.6f} ({mean * 100:.2f}%)")

        if lower is not None:
            print(
                f"  95%-й доверительный интервал: "
                f"[{max(0, lower):.6f}; "
                f"{min(1, upper):.6f}]"
            )

    normal_time = sum(stats[0] for stats in time_stats[:3])
    emergency_time = time_stats[3][0]

    print()
    print(f"Доля времени в нормальном режиме: "
          f"{normal_time * 100:.4f}%")
    print(f"Доля времени в аварийном режиме: "
          f"{emergency_time * 100:.4f}%")

    print()
    print("ДОПОЛНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ")

    print(
        f"Среднее количество переданных сообщений: "
        f"{message_stats[0]:.2f}"
    )

    print(
        f"Среднее количество сбоев: "
        f"{error_stats[0]:.2f}"
    )

    print(
        f"Среднее количество входов в аварию: "
        f"{emergency_stats[0]:.4f}"
    )

    print()
    print("ТЕОРЕТИЧЕСКИЕ ВЕРОЯТНОСТИ")
    print("S0: 0.900000")
    print("S1: 0.090000")
    print("S2: 0.009000")
    print("S3: 0.001000")
    print("Нормальный режим: 0.999000")
    print("Аварийный режим: 0.001000")


if __name__ == "__main__":
    main()
