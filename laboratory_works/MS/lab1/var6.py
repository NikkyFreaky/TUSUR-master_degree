
import random
import math
import statistics


# Исходные данные
SIMULATION_TIME = 240.0

MIN_ARRIVAL_TIME = 1.0
MAX_ARRIVAL_TIME = 20.0

SERVICE_TIME_1 = 8.0
SERVICE_TIME_2 = 14.0

PROBABILITY = 0.5


def generate_arrival_time():
    """Интервал между прибытиями от 1 до 20 минут."""
    u = random.random()

    return MIN_ARRIVAL_TIME + (
        MAX_ARRIVAL_TIME - MIN_ARRIVAL_TIME
    ) * u


def generate_service_time():
    """Время обслуживания: 8 или 14 минут."""
    u = random.random()

    if u < PROBABILITY:
        return SERVICE_TIME_1
    else:
        return SERVICE_TIME_2


def simulate_one():
    """Одна реализация работы магазина за 4 часа."""
    current_time = 0.0
    server_free_time = 0.0

    total_waiting_time = 0.0
    total_idle_time = 0.0

    customers_count = 0
    customers_waited = 0

    while True:
        # Время прибытия следующего покупателя
        current_time += generate_arrival_time()

        # Магазин закрыт для новых покупателей
        if current_time > SIMULATION_TIME:
            break

        customers_count += 1

        # Генерируем время обслуживания
        service_time = generate_service_time()

        if current_time >= server_free_time:
            # Сотрудник свободен: фиксируем простой
            idle_time = current_time - server_free_time
            total_idle_time += idle_time

            # Обслуживание начинается сразу
            service_start = current_time

        else:
            # Сотрудник занят: покупатель ожидает
            service_start = server_free_time

            waiting_time = service_start - current_time
            total_waiting_time += waiting_time
            customers_waited += 1

        # Время завершения обслуживания
        server_free_time = service_start + service_time

    # Учитываем простой до окончания рабочего дня
    if server_free_time < SIMULATION_TIME:
        total_idle_time += (
            SIMULATION_TIME -
            max(current_time - 0.0, server_free_time)
        ) if False else SIMULATION_TIME - server_free_time

    # Время работы после закрытия магазина
    overtime = max(
        0.0,
        server_free_time - SIMULATION_TIME
    )

    return (
        total_waiting_time,
        total_idle_time,
        customers_count,
        customers_waited,
        overtime
    )


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
    """Многократное моделирование работы магазина."""
    results = [[] for _ in range(5)]

    for _ in range(n):
        result = simulate_one()

        for i, value in enumerate(result):
            results[i].append(value)

    return [calculate_stats(values) for values in results]


def print_result(name, stats, unit):
    """Вывод результата моделирования."""
    mean, lower, upper = stats

    print(f"{name}: {mean:.4f} {unit}")

    if lower is not None:
        print(
            f"  95%-й доверительный интервал: "
            f"[{lower:.4f}; {upper:.4f}]"
        )


def main():
    print("Моделирование системы обслуживания магазина")
    print(f"Продолжительность работы: {SIMULATION_TIME} мин")
    print("Интервал прибытия покупателей: 1-20 мин")
    print("Время обслуживания: 8 или 14 мин")
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
        ("Суммарное время ожидания", "мин"),
        ("Время простоя системы", "мин"),
        ("Количество покупателей", "чел."),
        ("Количество ожидавших покупателей", "чел."),
        ("Время работы после закрытия", "мин")
    ]

    for (name, unit), stats in zip(names, results):
        print_result(name, stats, unit)

    print()
    print("ДОПОЛНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ")

    avg_wait, avg_idle, avg_count, avg_waited, _ = [
        result[0] for result in results
    ]

    if avg_count > 0:
        print(
            f"Среднее ожидание одного покупателя: "
            f"{avg_wait / avg_count:.4f} мин"
        )

        print(
            f"Доля покупателей с ожиданием: "
            f"{avg_waited / avg_count * 100:.2f}%"
        )

    print(
        f"Доля времени простоя системы: "
        f"{avg_idle / SIMULATION_TIME * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
