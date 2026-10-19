import random
import math
import statistics


# Исходные данные
MEAN_A = 50.0
SIGMA_A = 10.0

B_MIN = 15.0
B_MAX = 25.0

C_VALUES = (10, 20, 30, 40)
C_PROBS = (0.1, 0.25, 0.5, 0.15)


def normal(mean, sigma):
    """Нормальная величина по методу Бокса–Мюллера."""
    u1 = 1.0 - random.random()
    u2 = random.random()

    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(
        2.0 * math.pi * u2
    )

    return mean + sigma * z


def sample_b():
    """Равномерная величина на интервале (15, 25)."""
    u = random.random()
    return B_MIN + (B_MAX - B_MIN) * u


def sample_c():
    """Дискретная величина C."""
    u = random.random()
    cumulative = 0.0

    for value, probability in zip(C_VALUES, C_PROBS):
        cumulative += probability
        if u < cumulative:
            return value

    return C_VALUES[-1]


def sample_z():
    """Z = (B + C) * A. Величины независимы."""
    a = normal(MEAN_A, SIGMA_A)
    return (sample_b() + sample_c()) * a


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
    """Многократная генерация Z."""
    return calculate_stats([sample_z() for _ in range(n)])


def theoretical_mean():
    """E[Z] = (E[B] + E[C]) * E[A] при независимости."""
    mean_b = (B_MIN + B_MAX) / 2.0
    mean_c = sum(
        value * probability
        for value, probability in zip(C_VALUES, C_PROBS)
    )

    return (mean_b + mean_c) * MEAN_A


def main():
    print("Моделирование случайной величины Z = (B + C) * A")
    print(
        f"A: нормальный закон, m={MEAN_A:.0f}, "
        f"сигма={SIGMA_A:.0f}"
    )
    print(
        f"B: равномерно ({B_MIN:.0f}; {B_MAX:.0f})"
    )
    print(
        "C: "
        + ", ".join(
            f"{value} (p={probability})"
            for value, probability in zip(C_VALUES, C_PROBS)
        )
    )
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    mean, lower, upper = monte_carlo(n)
    expected = theoretical_mean()

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print(f"Среднее Z: {mean:.4f}")

    if lower is not None:
        print(
            "95%-й доверительный интервал: "
            f"[{lower:.4f}; {upper:.4f}]"
        )

    print()
    print("ТЕОРЕТИЧЕСКОЕ СРЕДНЕЕ")
    print(f"E[Z]: {expected:.4f}")


if __name__ == "__main__":
    main()
