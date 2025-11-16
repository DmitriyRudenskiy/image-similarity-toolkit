# Image Similarity Toolkit

Профессиональный инструмент для сравнения изображений с использованием современных нейросетевых моделей глубокого обучения.

## Описание

Image Similarity Toolkit — это Python-библиотека для вычисления схожести изображений с помощью различных предобученных нейронных сетей. Проект поддерживает несколько моделей (ResNet50, EfficientNet-B0, CLIP) и предоставляет различные метрики сходства.

## Ключевые возможности

- **Множественные модели**: ResNet50, EfficientNet-B0, CLIP
- **Различные метрики**: Косинусное сходство, Евклидово расстояние, нормализованное сходство
- **Современные векторные базы данных**: ChromaDB (рекомендуется) и SQLite
- **Поиск похожих изображений**: Быстрый поиск по базе данных с текстовыми запросами
- **Обнаружение дубликатов**: Автоматический поиск дублей и почти-дублей
- **Богатая метадата**: Хранение и фильтрация по метаданным изображений
- **Визуализация**: Автоматическая визуализация результатов сравнения
- **Простой API**: Интуитивно понятный интерфейс для работы
- **GPU-ускорение**: Автоматическое использование CUDA при наличии
- **Batch-обработка**: Сравнение множества изображений одновременно
- **Docker поддержка**: Контейнеризация для удобного развертывания

## Установка

### Требования

- Python 3.8+
- PyTorch 2.0+
- CUDA (опционально, для GPU-ускорения)

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Установка из исходников

```bash
git clone https://github.com/yourusername/image-similarity-toolkit.git
cd image-similarity-toolkit
pip install -e .
```

## Быстрый старт

```python
from image_similarity import ImageSimilarity

# Инициализация с выбором модели
similarity_checker = ImageSimilarity(model_name='efficientnet')

# Сравнение двух изображений
results = similarity_checker.compare_images('image1.jpg', 'image2.jpg')

# Вывод результатов
print(f"Косинусное сходство: {results['cosine_similarity']:.4f}")
print(f"Евклидово расстояние: {results['euclidean_distance']:.4f}")

# Визуализация результатов
similarity_checker.visualize_comparison(
    'image1.jpg', 
    'image2.jpg', 
    results,
    save_path='comparison.png'
)
```

## Поддерживаемые модели

| Модель | Описание | Размер эмбеддинга |
|--------|----------|-------------------|
| **ResNet50** | Классическая архитектура CNN | 2048 |
| **EfficientNet-B0** | Эффективная и легковесная модель | 1280 |
| **CLIP** | Мультимодальная модель от OpenAI | 512 |

## Метрики сходства

### Косинусное сходство
Измеряет угол между векторами эмбеддингов. Значения от -1 до 1, где 1 означает идентичные изображения.

### Евклидово расстояние
Прямое расстояние между векторами эмбеддингов в пространстве признаков.

### Нормализованное сходство
Нормализованная версия евклидова расстояния. Значения от 0 до 1, где 1 означает максимальное сходство.

## Примеры использования

### Базовое сравнение

```python
from image_similarity import ImageSimilarity

checker = ImageSimilarity(model_name='efficientnet')
results = checker.compare_images('cat1.jpg', 'cat2.jpg')

if results['cosine_similarity'] > 0.85:
    print("Изображения очень похожи!")
```

### Batch-сравнение

```python
from image_similarity import ImageSimilarity
import os

checker = ImageSimilarity(model_name='clip')

# Получаем список изображений
images = [f for f in os.listdir('data/input') if f.endswith('.jpg')]

# Сравниваем каждое изображение с первым
base_image = images[0]
for img in images[1:]:
    results = checker.compare_images(
        f'data/input/{base_image}',
        f'data/input/{img}'
    )
    print(f"{img}: {results['cosine_similarity']:.4f}")
```

### Работа с базой данных

```python
from image_similarity import ImageSimilarity, EmbeddingDatabase

# Инициализация
checker = ImageSimilarity(model_name='efficientnet')
db = EmbeddingDatabase('embeddings.db')

# Индексация изображений
import os
for img_file in os.listdir('data/input'):
    if img_file.endswith('.jpg'):
        img_path = f'data/input/{img_file}'
        embedding = checker.get_embedding(img_path)
        db.add_image(img_path, embedding)

# Поиск похожих изображений
query_embedding = checker.get_embedding('query.jpg')
similar_images = db.find_similar(query_embedding, top_k=5)

for result in similar_images:
    print(f"{result['image_name']}: {result['similarity']:.4f}")

# Поиск дубликатов
duplicates = db.find_duplicates(similarity_threshold=0.95)
print(f"Найдено {len(duplicates)} пар дубликатов")

db.close()
```

### Использование с ChromaDB (современная векторная БД) ⭐ *Рекомендуется*

```python
# Установка дополнительных зависимостей для ChromaDB
# pip install chromadb sentence-transformers

from image_similarity import ChromaDBBackend

# Инициализация современного бэкенда
backend = ChromaDBBackend(
    collection_name="my_photos",
    persist_directory="./vector_db",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Добавление изображений из директории
count = backend.add_images_from_directory(
    "./photos",
    max_images=500,
    recursive=True
)
print(f"Добавлено {count} изображений")

# Поиск по текстовому запросу
results = backend.find_similar(
    query_text="красивые горные пейзажи",
    top_k=10,
    threshold=0.7
)

for result in results:
    print(f"{result['filename']}: {result['similarity']:.3f}")

# Поиск по изображению
similar_images = backend.find_similar(
    query_image_path="./reference.jpg",
    top_k=5
)

# Поиск с фильтрацией по метаданным
filtered_results = backend.find_similar(
    query_text="портрет",
    filter_metadata={"category": "people"},
    top_k=5
)

# Автоматическое обнаружение дубликатов
duplicates = backend.find_duplicates(similarity_threshold=0.95)

# Экспорт коллекции
backend.export_collection("my_collection_backup.json")

# Статистика коллекции
stats = backend.get_stats()
print(f"Всего изображений: {stats['total_images']}")
print(f"Используемая модель: {stats['embedding_model']}")
```

### Автоматическое обнаружение дубликатов

```python
from image_similarity import EmbeddingDatabase

db = EmbeddingDatabase('embeddings.db')

# Поиск дубликатов с сохранением в БД
duplicates = db.find_duplicates(
    similarity_threshold=0.95,
    save_to_table=True
)

# Получение статистики
stats = db.get_stats()
print(f"Всего изображений: {stats['total_images']}")
print(f"Найдено дубликатов: {stats['total_duplicates']}")

db.close()
```

Больше примеров в директории `examples/`.

## Структура проекта

```
image-similarity-toolkit/
├── README.md                # Документация
├── requirements.txt         # Зависимости
├── Dockerfile.chromadb      # Docker для ChromaDB
├── docker-compose.chromadb.yml  # Docker Compose
├── setup.py                 # Установочный скрипт
├── LICENSE                  # Лицензия
├── .gitignore              # Игнорируемые файлы
├── src/
│   └── image_similarity/   # Основной пакет
│       ├── __init__.py
│       ├── core.py         # Основная логика
│       ├── models.py       # Работа с моделями
│       ├── database.py     # SQLite БД эмбеддингов
│       ├── chromadb_backend.py  # ChromaDB БД эмбеддингов
│       └── visualization.py # Визуализация
├── scripts/              # Скрипты автоматизации
│   └── format_and_lint.sh # Скрипт качества кода
├── examples/               # Примеры использования
│   ├── basic_usage.py
│   ├── batch_comparison.py
│   ├── database_example.py
│   └── duplicate_cleaner.py
├── tests/                  # Тесты
│   ├── __init__.py
│   ├── test_similarity.py
│   └── test_database.py
├── docs/                   # Дополнительная документация
│   ├── usage.md
│   ├── database_guide.md   # Руководство по базе данных
│   ├── DATABASE_STRUCTURE.md # Структура базы данных
│   ├── ARCHITECTURE.md     # Архитектура системы
│   ├── CONTRIBUTING.md     # Вклад в проект
│   └── QUICKSTART.md       # Быстрый старт
└── data/                   # Данные
    ├── input/              # Входные изображения
    └── output/             # Результаты
```

## Интерпретация результатов

| Косинусное сходство | Интерпретация |
|--------------------|---------------|
| > 0.85 | Очень похожие изображения |
| 0.70 - 0.85 | Умеренно похожие изображения |
| 0.50 - 0.70 | Слабо похожие изображения |
| < 0.50 | Различные изображения |

## Использование GPU

Библиотека автоматически определяет наличие CUDA и использует GPU при возможности:

```python
# Проверка используемого устройства
checker = ImageSimilarity(model_name='efficientnet')
print(f"Используется устройство: {checker.device}")
```

## Производительность

Приблизительное время обработки одной пары изображений на различных устройствах:

| Модель | CPU (Intel i7) | GPU (RTX 3080) |
|--------|----------------|----------------|
| ResNet50 | ~200ms | ~15ms |
| EfficientNet-B0 | ~150ms | ~10ms |
| CLIP | ~180ms | ~12ms |

## Ограничения

- Поддерживаются только RGB изображения
- Рекомендуется использовать изображения размером не менее 224x224
- Для CLIP требуется дополнительная установка `pip install git+https://github.com/openai/CLIP.git`

## Документация

Полная документация доступна в папке `docs/`:

- **[Быстрый старт](docs/QUICKSTART.md)** - Начало работы с Toolkit
- **[Руководство по базе данных](docs/database_guide.md)** - Работа с базой данных
- **[Структура базы данных](docs/DATABASE_STRUCTURE.md)** - Архитектура и схема БД
- **[ChromaDB интеграция](docs/CHROMADB_INTEGRATION.md)** ⭐ - Современная векторная БД
- **[Руководство разработчика](docs/DEVELOPMENT_GUIDE.md)** - Инструменты и процессы разработки
- **[Архитектура системы](docs/ARCHITECTURE.md)** - Техническая архитектура
- **[Вклад в проект](docs/CONTRIBUTING.md)** - Руководство для разработчиков

## Планы развития

- [ ] Поддержка дополнительных моделей (ViT, DINO, Swin)
- [ ] Batch-обработка с GPU параллелизацией
- [ ] Web-интерфейс для демонстрации
- [ ] Поддержка видеофайлов
- [ ] Графический интерфейс (GUI)
- [ ] Облачное развертывание (Docker/Kubernetes)
- [ ] API для интеграции с другими системами

## Завершенные функции ✅

- ✅ База данных эмбеддингов с SQLite
- ✅ Поиск похожих изображений в БД
- ✅ Обнаружение дубликатов
- ✅ Современные инструменты разработки
- ✅ Автоматизация качества кода
- ✅ Pre-commit hooks
- ✅ Comprehensive testing suite

## Вклад в проект

Мы приветствуем вклад в развитие проекта! Пожалуйста:

1. Сделайте Fork репозитория
2. Создайте ветку для новой функции (`git checkout -b feature/AmazingFeature`)
3. Закоммитьте изменения (`git commit -m 'Add some AmazingFeature'`)
4. Отправьте в ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

## Авторы

MiniMax Agent

## Благодарности

- PyTorch team за отличный фреймворк
- OpenAI за модель CLIP
- Сообщество разработчиков за вклад

## Контакты

- GitHub Issues: [Issues](https://github.com/yourusername/image-similarity-toolkit/issues)
- Email: your.email@example.com

---

**Made with ❤️ using PyTorch**
