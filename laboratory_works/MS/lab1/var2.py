
import random
import math


# Параметры моделирования
SIMULATION_TIME = 60
STEP_TIME = 10
P_EXCEED = 0.6

# Состояния устройства
S0 = 0  # Исходное состояние
S1 = 1  # Одно превышение порога
S2 = 2  # Цель обнаружена
S3 = 3  # Одно непревышение после обнаружения


def next_state(state):
    """Определение следующего состояния системы."""
    u = random.random()

    # Сигнал превысил порог
    if u < P_EXCEED:
        if state == S0:
            return S1
        elif state == S1:
            return S2
        elif state == S2:
            return S2
        elif state == S3:
            return S2

    # Сигнал не превысил порог
    else:
        if state == S0:
            return S0
        elif state == S1:
            return S0
        elif state == S2:
            return S3
        elif state == S3:
            return S0


def simulate_one():
    """Одна реализация работы устройства за 60 минут."""
    state = S0
    detected = False

    steps = SIMULATION_TIME // STEP_TIME

    for _ in range(steps):
        state = next_state(state)

        # Фиксируем факт обнаружения цели
        if state == S2:
            detected = True

    # Было ли обнаружение и состояние через час
    return detected, state == S2


def confidence_interval(p, n):
    """Приближённый 95%-й доверительный интервал."""
    se = math.sqrt(p * (1 - p) / n)

    lower = max(0.0, p - 1.96 * se)
    upper = min(1.0, p + 1.96 * se)

    return lower, upper


def monte_carlo(n):
    """Моделирование методом Монте-Карло."""
    detected_count = 0
    final_detected_count = 0

    for _ in range(n):
        detected, final_detected = simulate_one()

        if detected:
            detected_count += 1

        if final_detected:
            final_detected_count += 1

    # Вероятность обнаружения хотя бы раз за час
    p_detected = detected_count / n

    # Вероятность нахождения в S2 через час
    p_final = final_detected_count / n

    ci_detected = confidence_interval(p_detected, n)
    ci_final = confidence_interval(p_final, n)

    return p_detected, p_final, ci_detected, ci_final


def main():
    print("Моделирование устройства обнаружения цели")
    print(f"Продолжительность моделирования: "
          f"{SIMULATION_TIME} минут")
    print(f"Интервал смены состояний: {STEP_TIME} минут")
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    p_detected, p_final, ci_detected, ci_final = monte_carlo(n)

    print()
    print(f"Количество реализаций: {n}")

    print()
    print("Обнаружение цели хотя бы один раз за час:")
    print(f"Вероятность: {p_detected:.6f}")
    print(f"В процентах: {p_detected * 100:.2f}%")
    print(f"95%-й доверительный интервал: "
          f"[{ci_detected[0]:.6f}; {ci_detected[1]:.6f}]")

    print()
    print("Нахождение в состоянии обнаружения через час:")
    print(f"Вероятность: {p_final:.6f}")
    print(f"В процентах: {p_final * 100:.2f}%")
    print(f"95%-й доверительный интервал: "
          f"[{ci_final[0]:.6f}; {ci_final[1]:.6f}]")


if __name__ == "__main__":
    main()
