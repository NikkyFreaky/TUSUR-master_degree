import random
import math
from fractions import Fraction


# Исходные данные
DAYS = 5
BANKROLL = 10
TARGET = 3
PAYOUT = 8


def play_day():
    """Одна партия, пока разность гербов и решек не станет равна 3.

    На партию отложено 10 долларов, бросок стоит 1 доллар.
    Если партия завершена, игрок получает 8 долларов.
    """
    difference = 0
    flips = 0

    while flips < BANKROLL and difference < TARGET:
        flips += 1

        if random.random() < 0.5:
            difference += 1
        else:
            difference -= 1

    if difference == TARGET:
        return PAYOUT - flips

    return -BANKROLL


def simulate_week():
    """Пять рабочих дней. Выигрыш — суммарная прибыль больше нуля."""
    profit = 0

    for _ in range(DAYS):
        profit += play_day()

    return profit > 0


def monte_carlo(n):
    """Оценка вероятности методом Монте-Карло."""
    successful = 0

    for _ in range(n):
        if simulate_week():
            successful += 1

    p = successful / n
    se = math.sqrt(p * (1.0 - p) / n)

    lower = max(0.0, p - 1.96 * se)
    upper = min(1.0, p + 1.96 * se)

    return p, lower, upper, successful


def day_distribution():
    """Точные вероятности дневной прибыли."""
    states = {(0, 0): Fraction(1)}
    outcomes = {}

    for _ in range(BANKROLL):
        nxt = {}

        for (flips, difference), probability in states.items():
            for step in (1, -1):
                new_flips = flips + 1
                new_difference = difference + step
                branch = probability / 2

                if new_difference == TARGET:
                    profit = PAYOUT - new_flips
                    outcomes[profit] = (
                        outcomes.get(profit, Fraction(0)) + branch
                    )
                elif new_flips == BANKROLL:
                    outcomes[-BANKROLL] = (
                        outcomes.get(-BANKROLL, Fraction(0)) + branch
                    )
                else:
                    key = (new_flips, new_difference)
                    nxt[key] = nxt.get(key, Fraction(0)) + branch

        states = nxt

    return outcomes


def theoretical_probability():
    """Вероятность положительной прибыли за 5 дней."""
    outcomes = day_distribution()
    totals = {0: Fraction(1)}

    for _ in range(DAYS):
        nxt = {}

        for total, probability in totals.items():
            for profit, chance in outcomes.items():
                key = total + profit
                nxt[key] = nxt.get(key, Fraction(0)) + probability * chance

        totals = nxt

    return sum(
        probability
        for total, probability in totals.items()
        if total > 0
    )


def main():
    print("Моделирование игры «тройка»")
    print(f"Рабочих дней: {DAYS}")
    print(f"Деньги на игру: {BANKROLL} долл.")
    print(
        f"Партия заканчивается, когда гербов на {TARGET} "
        f"больше, чем решек"
    )
    print(
        f"Бросок стоит 1 долл., выигрыш партии — {PAYOUT} долл."
    )
    print("Монета симметричная")
    print()

    n = int(input("Введите количество реализаций: "))

    if n <= 0:
        print("Количество реализаций должно быть положительным.")
        return

    p, lower, upper, successful = monte_carlo(n)
    exact = theoretical_probability()

    print()
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print(f"Количество реализаций: {n}")
    print(f"Недель в выигрыше: {successful}")
    print(f"Вероятность: {p:.6f}")
    print(f"В процентах: {p * 100:.2f}%")
    print(
        "95%-й доверительный интервал: "
        f"[{lower:.6f}; {upper:.6f}]"
    )

    print()
    print("ТОЧНАЯ ВЕРОЯТНОСТЬ")
    print(f"Шанс в выигрыше за неделю: {float(exact):.6f}")
    print(f"В процентах: {float(exact) * 100:.2f}%")


if __name__ == "__main__":
    main()
