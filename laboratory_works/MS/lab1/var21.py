import random
import math


# Исходные данные
HITS_NEEDED = 5

# «18+6» и «20+5» секунд: равномерное время выстрела
SHOT_1 = (12.0, 24.0)
SHOT_2 = (15.0, 25.0)

# «5+2» секунды: перезарядка запасного патрона
RELOAD = (3.0, 7.0)

P_HIT_1 = 0.8
P_HIT_2 = 0.9
SPARES_1 = 2
SPARES_2 = 1


def uniform(left, right):
    """Равномерное время на отрезке [left, right]."""
    u = random.random()
    return left + (right - left) * u


def shooting_time(hit_probability, shot_bounds, spares):
    """Время пятого попадания. None, если патронов не хватило."""
    current_time = 0.0
    hits = 0
    ammo = HITS_NEEDED + spares
    shots = 0

    while hits < HITS_NEEDED and ammo > 0:
        ammo -= 1
        shots += 1

        # Запасной патрон заряжается только после основных пяти
        if shots > HITS_NEEDED:
            current_time += uniform(*RELOAD)

        current_time += uniform(*shot_bounds)

        if random.random() < hit_probability:
            hits += 1

    if hits < HITS_NEEDED:
        return None

    return current_time


def simulate_one():
    """Первый стрелок набрал пять попаданий строго раньше второго."""
    time_1 = shooting_time(P_HIT_1, SHOT_1, SPARES_1)
    time_2 = shooting_time(P_HIT_2, SHOT_2, SPARES_2)

    first_failed = time_1 is None
    second_failed = time_2 is None
    first_earlier = time_1 is not None and (
        time_2 is None or time_1 < time_2
    )

    return first_earlier, first_failed, second_failed


def monte_carlo(n):
    """Оценка вероятности методом Монте-Карло."""
    earlier = 0
    first_failed = 0
    second_failed = 0

    for _ in range(n):
        is_earlier, fail_1, fail_2 = simulate_one()

        if is_earlier:
            earlier += 1
        if fail_1:
            first_failed += 1
        if fail_2:
            second_failed += 1

    p = earlier / n
    se = math.sqrt(p * (1.0 - p) / n)
    lower = max(0.0, p - 1.96 * se)
    upper = min(1.0, p + 1.96 * se)

    return (
        p,
        lower,
        upper,
        earlier,
        first_failed / n,
        second_failed / n
    )


def failure_probability(hit_probability, spares):
    """Вероятность меньше пяти попаданий при всех патронах."""
    trials = HITS_NEEDED + spares
    probability = 0.0

    for hits in range(HITS_NEEDED):
        probability += (
            math.comb(trials, hits)
            * hit_probability ** hits
            * (1.0 - hit_probability) ** (trials - hits)
        )

    return probability


def main():
    print("Моделирование стрельбы двух стрелков")
    print(f"Нужно попаданий: {HITS_NEEDED}")
    print(
        f"Первый: p={P_HIT_1}, выстрел "
        f"[{SHOT_1[0]:.0f}; {SHOT_1[1]:.0f}] с, "
        f"запасных патронов {SPARES_1}"
    )
    print(
        f"Второй: p={P_HIT_2}, выстрел "
        f"[{SHOT_2[0]:.0f}; {SHOT_2[1]:.0f}] с, "
        f"запасных патронов {SPARES_2}"
    )
    print(
        f"Перезарядка: [{RELOAD[0]:.0f}; {RELOAD[1]:.0f}] с"
    )
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    (
        p,
        lower,
        upper,
        earlier,
        fail_1,
        fail_2
    ) = monte_carlo(n)

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print(f"Первый успел раньше: {earlier}")
    print(f"Вероятность: {p:.6f}")
    print(f"В процентах: {p * 100:.2f}%")
    print(
        "95%-й доверительный интервал: "
        f"[{lower:.6f}; {upper:.6f}]"
    )

    print()
    print("ДОЛЯ СЕРИЙ БЕЗ ПЯТИ ПОПАДАНИЙ")
    print(f"Первый стрелок: {fail_1:.6f}")
    print(f"Второй стрелок: {fail_2:.6f}")

    print()
    print("ТОЧНЫЕ ВЕРОЯТНОСТИ НЕ ДОБРАТЬ ПЯТЬ ПОПАДАНИЙ")
    print(
        f"Первый стрелок: "
        f"{failure_probability(P_HIT_1, SPARES_1):.6f}"
    )
    print(
        f"Второй стрелок: "
        f"{failure_probability(P_HIT_2, SPARES_2):.6f}"
    )


if __name__ == "__main__":
    main()
