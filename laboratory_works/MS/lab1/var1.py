
import random
import math


def exponential(mean):
    """Генерация экспоненциального времени работы."""
    u = random.random()
    return -mean * math.log(1 - u)


def replacement_time():
    """Время замены узла A: от 2 до 4 минут."""
    return 2 + 2 * random.random()


def simulate_one():
    """Одна реализация работы устройства за 120 минут."""
    t = 0.0
    end_time = 120.0

    spares = 2
    b_working = True

    # Время до отказа вспомогательного узла B
    b_remaining = exponential(60)

    while t < end_time:

        # Среднее время работы A зависит от B
        mean_a = 40 if b_working else 20

        # Время до очередного отказа A
        a_remaining = exponential(mean_a)

        if b_working and b_remaining < a_remaining:
            # Первым выходит из строя узел B
            t += b_remaining
            b_working = False

            if t >= end_time:
                return True

            # A продолжает работать в ухудшенном режиме.
            # Благодаря свойству отсутствия памяти
            # генерируем новое время до отказа A.
            continue

        # Первым выходит из строя узел A
        if t + a_remaining >= end_time:
            return True

        t += a_remaining

        # Учитываем время эксплуатации узла B
        if b_working:
            b_remaining -= a_remaining

        # Нет запасных узлов A: окончательный отказ
        if spares == 0:
            return False

        # Замена вышедшего из строя узла A
        spares -= 1
        t += replacement_time()

        # Если в момент 120 минут идёт замена,
        # устройство находится в нерабочем состоянии
        if t >= end_time:
            return False

    return True


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

    return p, lower, upper


def main():
    print("Моделирование надёжности устройства")
    print("Продолжительность работы: 120 минут")
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    p, lower, upper = monte_carlo(n)

    print()
    print(f"Количество реализаций: {n}")
    print(f"Вероятность работоспособности: {p:.6f}")
    print(f"В процентах: {p * 100:.2f}%")
    print(f"95%-й доверительный интервал: "
          f"[{lower:.6f}; {upper:.6f}]")


if __name__ == "__main__":
    main()
