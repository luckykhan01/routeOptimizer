# routeOptimizer

ML-система маршрутизации (VRPTW), прогнозирующая реальное время в пути (ETA) для оптимизации доставки
Обучена на реальных логах поездок **NYC TLC Yellow Taxi (Январь 2023)**.

![demo](docs/demo.gif)

## Architecture

```mermaid
graph TD
    DataGen["Генерация данных<br/><code>src/data/generate_data.py</code>"] --> FeatEng["Feature Engineering<br/><code>src/features/engineering.py</code>"]
    FeatEng --> ETAModel["ETA модель (LightGBM)<br/><code>src/models/train_eta.py</code>"]
    ETAModel --> TimeMatrix["Матрица времени<br/><code>src/optimization/solver.py</code>"]
    TimeMatrix --> Solver["VRPTW солвер (OR-Tools)<br/><code>src/optimization/solver.py</code>"]
    Solver --> FastAPI["FastAPI<br/><code>src/api/main.py</code>"]
    FastAPI --> Streamlit["Streamlit UI<br/><code>src/ui/app.py</code>"]
```

## Quick Start

```bash
docker-compose up --build
```

## Результаты (120 заказов, 3 курьера)

| baseline_type | MAE (sec) | total_time_sec | served_orders | dropped_orders |
| :--- | :--- | :--- | :--- | :--- |
| **ml** (LightGBM) | ~150 | 26543 | 77 | 43 |
| **median** speed | — | 16435 | 90 | 30 |
| **constant** speed | — | 14402 | 95 | 25 |

> **Insight:** Базовые эвристики (constant/median) обещают развезти больше заказов (90-95), опираясь на идеальную скорость по прямой. ML-модель обученная на реальных логах NYC TLC, оценивает время с учетом городского трафика и снижает план до реалистичных 77 заказов и предотвращает массовые опоздания курьеров.

## Структура проекта

```text
.
├── configs/          # конфигурации обучения
├── data/             # сырые/обработанные данные
├── models/           # веса ETA модели
├── scripts/          # скрипты автоматизации
├── src/              
│   ├── api/          # FastAPI эндпоинты
│   ├── data/         # пайплайн данных NYC TLC
│   ├── features/     # генерация признаков (haversine, cyc_time)
│   ├── models/       # LightGBM пайплайн
│   ├── optimization/ # OR-Tools VRPTW солвер
│   └── ui/           # Streamlit дашборд
└── tests/            # pytest
```

## Стек технологий

Python, LightGBM, Google OR-Tools, FastAPI, Streamlit, Pandas, OSMnx, Docker

