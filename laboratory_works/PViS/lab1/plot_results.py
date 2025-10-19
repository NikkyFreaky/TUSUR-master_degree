from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# читаем CSV, созданный benchmark.py
df = pd.read_csv("results.csv")

# убеждаемся, что нужные колонки есть
required_cols = {"video", "processes", "run", "elapsed_time"}
if not required_cols.issubset(df.columns):
    raise ValueError(
        f"Missing required columns in CSV: {required_cols - set(df.columns)}"
    )

# создаем выходную папку
Path("plots").mkdir(exist_ok=True)

# группируем по числу процессов и вычисляем среднее
summary = df.groupby("processes", as_index=False)["elapsed_time"].mean()

# === ГРАФИК 1: среднее время выполнения ===
plt.figure(figsize=(8, 5))
sns.lineplot(data=summary, x="processes", y="elapsed_time", marker="o")
plt.title("Average Processing Time vs Number of Processes")
plt.xlabel("Number of Processes")
plt.ylabel("Average Time (s)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/avg_time_vs_processes.png", dpi=200)
plt.close()

# === ГРАФИК 2: Boxplot распределения времени ===
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="processes", y="elapsed_time")
plt.title("Execution Time Distribution by Process Count")
plt.xlabel("Number of Processes")
plt.ylabel("Execution Time (s)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("plots/boxplot_time_distribution.png", dpi=200)
plt.close()

print("✅ Plots saved to 'plots/' folder:")
print(" - avg_time_vs_processes.png")
print(" - boxplot_time_distribution.png")
