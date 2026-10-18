
import random
import math
import statistics


# Параметры случайных отклонений
SIGMA_X = 60.0
SIGMA_Y = 30.0

# Математические ожидания
MEAN_X = 0.0
MEAN_Y = 0.0

# Количество наблюдений в одной серии
OBSERVATIONS = 6


def normal_random(mean, sigma):
    """Нормальное распределение методом Бокса — Мюллера."""
    u1 = 1.0 - random.random()
    u2 = random.random()

    z = math.sqrt(-2.0 * math.log(u1))
    z *= math.cos(2.0 * math.pi * u2)

    return mean + sigma * z


def simulate_one():
    """Одна серия из шести случайных наблюдений."""
    coordinates = []

    for _ in range(OBSERVATIONS):
        x = normal_random(MEAN_X, SIGMA_X)
        y = normal_random(MEAN_Y, SIGMA_Y)

        coordinates.append((x, y))

    return coordinates


def monte_carlo(n):
    """Многократное моделирование случайных отклонений."""
    all_x = []
    all_y = []

    for _ in range(n):
        coordinates = simulate_one()

        for x, y in coordinates:
            all_x.append(x)
            all_y.append(y)

    mean_x = statistics.mean(all_x)
    mean_y = statistics.mean(all_y)

    std_x = statistics.stdev(all_x)
    std_y = statistics.stdev(all_y)

    total = len(all_x)

    ci_x = 1.96 * std_x / math.sqrt(total)
    ci_y = 1.96 * std_y / math.sqrt(total)

    return mean_x, mean_y, std_x, std_y, ci_x, ci_y


def main():
    print("Моделирование двумерных случайных отклонений")
    print(f"Среднеквадратическое отклонение X: {SIGMA_X} м")
    print(f"Среднеквадратическое отклонение Y: {SIGMA_Y} м")
    print(f"Наблюдений в одной серии: {OBSERVATIONS}")
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    mean_x, mean_y, std_x, std_y, ci_x, ci_y = monte_carlo(n)

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print(f"Общее количество наблюдений: {n * OBSERVATIONS}")
    print()

    print(f"Среднее отклонение X: {mean_x:.4f} м")
    print(f"Среднее отклонение Y: {mean_y:.4f} м")
    print(f"Стандартное отклонение X: {std_x:.4f} м")
    print(f"Стандартное отклонение Y: {std_y:.4f} м")

    print()
    print("95%-е доверительные интервалы для средних:")
    print(f"X: [{mean_x - ci_x:.4f}; {mean_x + ci_x:.4f}]")
    print(f"Y: [{mean_y - ci_y:.4f}; {mean_y + ci_y:.4f}]")


if __name__ == "__main__":
    main()
