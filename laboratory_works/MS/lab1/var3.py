
import random
import math


# Исходные данные
BLOCKS = 1000
MEAN_TRANSMISSION = 5.0
MEAN_CORRECTION = 1.0


def exponential(mean):
    """Генерация экспоненциально распределённого времени."""
    u = random.random()
    return -mean * math.log(1 - u)


def generate_error(p0, p1, p2, p3):
    """Определение типа ошибки с заданными вероятностями."""
    u = random.random()

    if u < p0:
        return 0  # Ошибок нет
    elif u < p0 + p1:
        return 1  # Одиночная ошибка
    elif u < p0 + p1 + p2:
        return 2  # Двойная ошибка
    else:
        return 3  # Тройная ошибка


def simulate_one(p0, p1, p2, p3):
    """Передача 1000 блоков информации."""
    correction_time = 0.0
    retransmission_time = 0.0
    total_time = 0.0

    corrections = 0
    retransmissions = 0

    for _ in range(BLOCKS):
        first_attempt = True

        while True:
            # Генерируем время передачи блока
            t = exponential(MEAN_TRANSMISSION)
            total_time += t

            # Учитываем время повторных передач отдельно
            if not first_attempt:
                retransmission_time += t
                retransmissions += 1

            first_attempt = False

            # Определяем возникшую ошибку
            error = generate_error(p0, p1, p2, p3)

            if error == 0:
                # Блок успешно передан
                break

            elif error == 1:
                # Исправление одиночной ошибки
                t_corr = exponential(MEAN_CORRECTION)

                correction_time += t_corr
                total_time += t_corr
                corrections += 1
                break

            else:
                # Двойная или тройная ошибка:
                # передаём тот же блок заново
                continue

    extra_time = correction_time + retransmission_time

    return (
        correction_time,
        retransmission_time,
        extra_time,
        total_time,
        corrections,
        retransmissions
    )


def calculate_stats(values):
    """Среднее и 95%-й доверительный интервал."""
    n = len(values)
    mean = sum(values) / n

    if n == 1:
        return mean, None, None

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(variance / n)

    lower = mean - 1.96 * se
    upper = mean + 1.96 * se

    return mean, lower, upper


def monte_carlo(n, p0, p1, p2, p3):
    """Повторное моделирование системы."""
    results = [[] for _ in range(6)]

    for _ in range(n):
        result = simulate_one(p0, p1, p2, p3)

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
    print("Моделирование системы передачи данных")
    print(f"Количество блоков: {BLOCKS}")
    print(f"Среднее время передачи: {MEAN_TRANSMISSION} с")
    print(f"Среднее время коррекции: {MEAN_CORRECTION} с")
    print()

    print("Введите вероятности возникновения ошибок:")

    p0 = float(input("Без ошибок (p0): "))
    p1 = float(input("Одиночная ошибка (p1): "))
    p2 = float(input("Двойная ошибка (p2): "))
    p3 = float(input("Тройная ошибка (p3): "))

    probabilities = [p0, p1, p2, p3]

    if any(p < 0 or p > 1 for p in probabilities):
        print("Ошибка: вероятности должны быть от 0 до 1.")
        return

    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        print("Ошибка: сумма вероятностей должна быть равна 1.")
        return

    if p0 + p1 <= 0:
        print("Ошибка: успешная передача невозможна.")
        return

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    results = monte_carlo(n, p0, p1, p2, p3)

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print(f"Блоков в каждой реализации: {BLOCKS}")
    print()

    names = [
        ("Время коррекции ошибок", "с"),
        ("Время повторной передачи", "с"),
        ("Общие дополнительные затраты", "с"),
        ("Общее время передачи информации", "с"),
        ("Количество коррекций", "шт."),
        ("Количество повторных передач", "шт.")
    ]

    for (name, unit), stats in zip(names, results):
        print_result(name, stats, unit)


if __name__ == "__main__":
    main()
