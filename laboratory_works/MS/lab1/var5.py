import random 
import math

mean_x = 0
mean_y = 0

sigma_x = 60
sigma_y = 30

factory_x = 75
factory_y = 30

rockets = 6

def normal_dist():
  while True:
    u1 = 2 * random.random() - 1
    u2 = 2 * random.random() - 1
    s = u1 ** 2 + u2 ** 2

    if 0 < s < 1:
      multiplier = math.sqrt(-2 * math.log(s) / s)
      z1 = u1 * multiplier
      z2 = u2 * multiplier

      return z1, z2

def simulation():
  coord = []

  for i in range(rockets):
    z1, z2 = normal_dist()
    x = mean_x + sigma_x * z1
    y = mean_y + sigma_y * z2

    coord.append((x, y))

  return coord

def is_hit(x, y):
  return abs(x) <= factory_x and abs(y) <= factory_y

def monte_carlo(n):
  total_hits = 0

  for i in range(n):
    coord = simulation()

    for x,y in coord:
      if is_hit(x, y):
        total_hits += 1

  return total_hits / n


def main():
  n = int(input("Введите количество выполнений: "))
  average_hits = monte_carlo(n)
  print(f"\nСреднее число попаданий при атаке: {average_hits:.4f}\n")

if __name__ == "__main__":
    main()
