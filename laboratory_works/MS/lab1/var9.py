
import random
import math
import statistics


# Исходные данные
SIMULATION_TIME = 30 * 24  # 720 часов

ARRIVAL_RATE = 4.0         # Автомобиля в час
SERVICE_RATE = 3.0         # Автомобиля в час


def exponential(rate):
    """Генерация экспоненциального времени."""
    u = random.random()
    return -math.log(1 - u) / rate


def simulate_one():
    """Одна реализация работы бензоколонки за 30 суток."""
    current_time = 0.0
    server_free_time = 0.0

    arrived = 0
    accepted = 0
    served = 0
    rejected = 0

    while True:
        # Время прибытия следующего автомобиля
        current_time += exponential(ARRIVAL_RATE)

        # Период моделирования завершён
        if current_time > SIMULATION_TIME:
            break

        arrived += 1

        if current_time >= server_free_time:
            # Бензоколонка свободна
            accepted += 1

            service_time = exponential(SERVICE_RATE)
            server_free_time = current_time + service_time

            # Обслуживание завершилось в пределах 30 суток
            if server_free_time <= SIMULATION_TIME:
                served += 1

        else:
            # Бензоколонка занята: автомобиль уезжает
            rejected += 1

    # Принят, но ещё обслуживается на конец периода
    unfinished = accepted - served

    return arrived, accepted, served, rejected, unfinished


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
    """Многократное моделирование методом Монте-Карло."""
    results = [[] for _ in range(5)]

    for _ in range(n):
        result = simulate_one()

        for i, value in enumerate(result):
            results[i].append(value)

    return [calculate_stats(values) for values in results]


def print_result(name, stats, unit):
    """Вывод значения и доверительного интервала."""
    mean, lower, upper = stats

    print(f"{name}: {mean:.4f} {unit}")

    if lower is not None:
        print(
            f"  95%-й доверительный интервал: "
            f"[{lower:.4f}; {upper:.4f}]"
        )


def main():
    print("Моделирование работы бензозаправочной станции")
    print(f"Продолжительность: {SIMULATION_TIME} часов")
    print(f"Интенсивность прибытия: {ARRIVAL_RATE} авто/час")
    print(f"Интенсивность обслуживания: {SERVICE_RATE} авто/час")
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    results = monte_carlo(n)

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    names = [
        ("Всего прибывших автомобилей", "шт."),
        ("Принято на обслуживание", "шт."),
        ("Полностью обслужено автомобилей", "шт."),
        ("Отказано автомобилям", "шт."),
        ("Не завершили обслуживание", "шт.")
    ]

    for (name, unit), stats in zip(names, results):
        print_result(name, stats, unit)

    avg_arrived = results[0][0]
    avg_accepted = results[1][0]
    avg_served = results[2][0]
    avg_rejected = results[3][0]

    print()
    print("ДОПОЛНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ")

    if avg_arrived > 0:
        print(
            f"Доля принятых автомобилей: "
            f"{avg_accepted / avg_arrived * 100:.2f}%"
        )

        print(
            f"Доля отказов: "
            f"{avg_rejected / avg_arrived * 100:.2f}%"
        )

    print()
    print("ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ")

    p_free = SERVICE_RATE / (ARRIVAL_RATE + SERVICE_RATE)
    p_busy = ARRIVAL_RATE / (ARRIVAL_RATE + SERVICE_RATE)

    theoretical_arrivals = ARRIVAL_RATE * SIMULATION_TIME

    # Учитываем, что в начале моделирования колонка свободна
    k = ARRIVAL_RATE + SERVICE_RATE

    theoretical_accepted = (
        ARRIVAL_RATE * p_free * SIMULATION_TIME
        + ARRIVAL_RATE * p_busy / k
        * (1 - math.exp(-k * SIMULATION_TIME))
    )

    theoretical_rejected = (
        theoretical_arrivals - theoretical_accepted
    )

    theoretical_unfinished = (
        p_busy * (1 - math.exp(-k * SIMULATION_TIME))
    )

    theoretical_served = (
        theoretical_accepted - theoretical_unfinished
    )

    print(f"Прибывших: {theoretical_arrivals:.4f}")
    print(f"Принятых: {theoretical_accepted:.4f}")
    print(f"Обслуженных: {theoretical_served:.4f}")
    print(f"Отказанных: {theoretical_rejected:.4f}")

    print()
    print("ОТКЛОНЕНИЕ ОТ ТЕОРИИ")

    print(
        f"Обслуженные: "
        f"{abs(avg_served - theoretical_served):.4f} авто"
    )

    print(
        f"Отказанные: "
        f"{abs(avg_rejected - theoretical_rejected):.4f} авто"
    )


if __name__ == "__main__":
    main()
