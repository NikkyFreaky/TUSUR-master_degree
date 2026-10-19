import random
import math


# Исходные данные
BLOCKS = 10
MAX_DISTANCE = 2


def generate_step():
    """Шаг на перекрёстке: север, юг, восток или запад."""
    u = random.random()

    if u < 0.25:
        return 0, 1

    if u < 0.50:
        return 0, -1

    if u < 0.75:
        return 1, 0

    return -1, 0


def simulate_one():
    """Одна прогулка на десять кварталов.

    Расстояние до дома измеряется в кварталах по улицам:
    |x| + |y|. На целочисленной сетке это то же множество
    точек, что и круг радиуса 2.
    """
    x = 0
    y = 0

    for _ in range(BLOCKS):
        dx, dy = generate_step()
        x += dx
        y += dy

    return abs(x) + abs(y) <= MAX_DISTANCE


def monte_carlo(n):
    """Оценка вероятности методом Монте-Карло."""
    successful = 0

    for _ in range(n):
        if simulate_one():
            successful += 1

    p = successful / n

    # Стандартная ошибка оценки вероятности
    se = math.sqrt(p * (1 - p) / n)

    # Приближённый 95%-й доверительный интервал
    lower = max(0.0, p - 1.96 * se)
    upper = min(1.0, p + 1.96 * se)

    return p, lower, upper, successful


def main():
    print("Моделирование случайного блуждания")
    print(f"Число пройденных кварталов: {BLOCKS}")
    print(
        "На каждом перекрёстке выбор направления "
        "равновероятен: север, юг, восток, запад"
    )
    print(
        f"Условие успеха: расстояние до дома "
        f"не больше {MAX_DISTANCE} кварталов"
    )
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    p, lower, upper, successful = monte_carlo(n)

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print(f"Удачных прогулок: {successful}")
    print(f"Вероятность: {p:.6f}")
    print(f"В процентах: {p * 100:.2f}%")
    print(
        "95%-й доверительный интервал: "
        f"[{lower:.6f}; {upper:.6f}]"
    )


if __name__ == "__main__":
    main()
