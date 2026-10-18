
import random
import math
import heapq
import statistics


# Исходные данные
INITIAL_STOCK = 10000
SAFETY_STOCK = 200

WORKSHOPS = 3
BATCH_SIZE = 100

CONSUMPTION_RATE = 20.0


def exponential(rate, rng):
    """Время использования одного комплекта в сутках."""
    u = rng.random()
    return -math.log1p(-u) / rate


def simulate_one(rng):
    """Одна реализация работы склада."""

    # Выдача первоначальных партий
    stock = INITIAL_STOCK - WORKSHOPS * BATCH_SIZE

    # Остаток комплектов в каждом цехе
    workshop_stock = [BATCH_SIZE] * WORKSHOPS

    # Счётчик выполненных заявок
    supplied_batches = 0

    # Очередь событий: (время, номер цеха)
    events = []

    for i in range(WORKSHOPS):
        next_time = exponential(CONSUMPTION_RATE, rng)
        heapq.heappush(events, (next_time, i))

    while True:
        # Выбираем ближайшее событие расходования
        current_time, workshop = heapq.heappop(events)

        # Один комплект использован
        workshop_stock[workshop] -= 1

        # Если комплектующие закончились, нужна заявка
        if workshop_stock[workshop] == 0:

            # Проверяем возможность выдачи новой партии
            if stock - BATCH_SIZE < SAFETY_STOCK:
                # Запас недостаточен:
                # фиксируем время завершения
                return (
                    current_time,
                    supplied_batches,
                    stock
                )

            # Склад удовлетворяет заявку
            stock -= BATCH_SIZE
            supplied_batches += 1

            workshop_stock[workshop] = BATCH_SIZE

        # Планируем использование следующего комплекта
        next_time = current_time + exponential(
            CONSUMPTION_RATE, rng
        )

        heapq.heappush(events, (next_time, workshop))


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


def monte_carlo(n, seed=None):
    """Многократное моделирование работы склада."""

    rng = random.Random(seed)

    times = []
    supplied = []
    remaining_stock = []

    for _ in range(n):
        time, batches, stock = simulate_one(rng)

        times.append(time)
        supplied.append(batches)
        remaining_stock.append(stock)

    return (
        calculate_stats(times),
        calculate_stats(supplied),
        calculate_stats(remaining_stock)
    )


def print_result(name, stats, unit):
    """Вывод среднего значения и доверительного интервала."""
    mean, lower, upper = stats

    print(f"{name}: {mean:.4f} {unit}")

    if lower is not None:
        print(
            f"  95%-й доверительный интервал: "
            f"[{lower:.4f}; {upper:.4f}]"
        )


def main():
    print("Моделирование работы склада комплектующих")
    print(f"Нормативный запас: {INITIAL_STOCK} комплектов")
    print(f"Страховой запас: {SAFETY_STOCK} комплектов")
    print(f"Количество цехов: {WORKSHOPS}")
    print(f"Размер партии: {BATCH_SIZE} комплектов")
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    results = monte_carlo(n)

    time_stats, batches_stats, stock_stats = results

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print()

    print_result(
        "Время до исчерпания доступного запаса",
        time_stats,
        "суток"
    )

    print_result(
        "Количество выполненных дополнительных заявок",
        batches_stats,
        "шт."
    )

    print_result(
        "Остаток деталей на складе",
        stock_stats,
        "комплектов"
    )

    avg_time = time_stats[0]

    print()
    print("ДОПОЛНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ")

    print(f"Среднее время в часах: {avg_time * 24:.2f} ч")

    print(
        f"Среднее время в месяцах (по 30 дней): "
        f"{avg_time / 30:.2f}"
    )

    # Приближённая аналитическая оценка
    available = (
        INITIAL_STOCK
        - SAFETY_STOCK
        - WORKSHOPS * BATCH_SIZE
    )

    theoretical = available / (
        WORKSHOPS * CONSUMPTION_RATE
    )

    print()
    print("АНАЛИТИЧЕСКАЯ ОЦЕНКА")
    print(f"Расчётное время: {theoretical:.4f} суток")

    difference = abs(avg_time - theoretical)

    print(f"Абсолютное отклонение: {difference:.4f} суток")


if __name__ == "__main__":
    main()
