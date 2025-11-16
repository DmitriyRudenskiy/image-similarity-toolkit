# Image Similarity Toolkit - DDD Architecture

## Обзор

Этот проект был переработан с использованием принципов **Domain Driven Design (DDD)** для создания более чистой, модульной и поддерживаемой архитектуры. DDD помогает создать код, который точно отражает предметную область и легко адаптируется к изменениям.

## Основные принципы DDD в проекте

### 🏗️ Слоистая архитектура

```
src_ddd/
├── domain/           # Ядро бизнес-логики
├── application/      # Сценарии использования
├── infrastructure/   # Внешние зависимости
├── interfaces/       # Пользовательские интерфейсы
└── shared/           # Общие компоненты
```

### 🎯 Ключевые концепции

#### 1. **Bounded Contexts (Ограниченные контексты)**
- **Image Processing** - обработка изображений
- **Vector Storage** - хранение векторных представлений  
- **Similarity Search** - поиск похожести
- **Database Management** - управление базами данных
- **Configuration** - конфигурация системы

#### 2. **Value Objects (Объекты-значения)**
Неизменяемые объекты, которые определяются своими атрибутами:
- `Image` - представление изображения
- `VectorEmbedding` - векторное представление
- `SimilarityQuery` - запрос поиска
- `SimilarityResult` - результат поиска
- `Configuration` - конфигурация системы

#### 3. **Aggregate Roots (Корни агрегатов)**
Главные объекты, которые координируют связанные сущности:
- `VectorStore` - управление векторным хранилищем

#### 4. **Domain Services (Сервисы домена)**
Логика, которая не принадлежит конкретным объектам:
- `SimilarityCalculator` - расчёт похожести
- `DuplicateDetector` - обнаружение дубликатов

#### 5. **Repository Pattern (Паттерн репозитория)**
Абстракция доступа к данным:
- `VectorRepository` - интерфейс для работы с векторами

## 🚀 Примеры использования

### Базовое использование

```python
from src_ddd.domain.configuration import Configuration, ModelConfiguration
from src_ddd.application.use_cases import AddImageUseCase
from src_ddd.infrastructure.database import SQLiteRepository

# 1. Настройка конфигурации
config = Configuration.default()
model_config = ModelConfiguration.efficientnet_b0()

# 2. Создание инфраструктуры
repository = SQLiteRepository(config.database)
vector_store = VectorStore(repository)

# 3. Использование use case
add_use_case = AddImageUseCase(vector_store, image_processor, embedding_generator)
request = AddImageRequest(image_path, model_config)
response = add_use_case.execute(request)
```

### Поиск похожих изображений

```python
from src_ddd.application.use_cases import SearchSimilarImagesUseCase
from src_ddd.domain.similarity_search import SimilarityQuery

# Создание поискового запроса
query = SimilarityQuery.from_image(
    image_path=Path("query.jpg"),
    limit=10,
    threshold=0.8
)

# Выполнение поиска
search_use_case = SearchSimilarImagesUseCase(vector_store, image_processor, embedding_generator)
results = search_use_case.execute(SearchSimilarImagesRequest.from_image(Path("query.jpg")))

print(f"Найдено {results.total_found} похожих изображений")
```

### Пакетная обработка

```python
from src_ddd.application.use_cases import BatchProcessImagesUseCase

# Пакетная обработка множества изображений
batch_request = BatchProcessImagesRequest(
    image_paths=[Path("img1.jpg"), Path("img2.jpg"), Path("img3.jpg")],
    model_config=ModelConfiguration.resnet50(),
    max_workers=4,
    fail_fast=False
)

batch_use_case = BatchProcessImagesUseCase(add_use_case, duplicate_detector)
response = batch_use_case.execute(batch_request)

print(f"Обработано: {response.successful_count}/{response.total_count}")
```

### Обнаружение дубликатов

```python
from src_ddd.application.use_cases import FindDuplicatesUseCase

# Поиск дубликатов
duplicate_request = FindDuplicatesRequest(
    threshold=0.95,
    group_similar=True,
    min_group_size=2
)

duplicate_use_case = FindDuplicatesUseCase(vector_store, duplicate_detector)
response = duplicate_use_case.execute(duplicate_request)

print(f"Найдено {len(response.duplicate_groups)} групп дубликатов")
```

## 📁 Структура проекта

### Domain Layer (Ядро домена)

```
domain/
├── image_processing/
│   ├── __init__.py
│   └── image.py           # Image Value Object
├── vector_storage/
│   ├── __init__.py
│   ├── vector_embedding.py    # VectorEmbedding Value Object
│   ├── vector_repository.py   # Repository Interface
│   └── vector_store.py        # Aggregate Root
├── similarity_search/
│   ├── __init__.py
│   ├── similarity_query.py    # SimilarityQuery Value Object
│   ├── similarity_result.py   # SimilarityResult Value Object
│   ├── similarity_calculator.py   # Domain Service
│   └── duplicate_detector.py      # Domain Service
├── database_management/
│   ├── __init__.py
│   ├── database_configuration.py  # Configuration Value Object
│   ├── database_connection.py     # Connection Interface
│   └── repository.py              # Base Repository Class
└── configuration/
    ├── __init__.py
    ├── configuration.py           # Configuration Value Object
    └── model_configuration.py     # ModelConfiguration Value Object
```

### Application Layer (Слой приложения)

```
application/
├── use_cases/
│   ├── __init__.py
│   ├── add_image_use_case.py          # Add Image Use Case
│   ├── search_similar_images_use_case.py  # Search Use Case
│   ├── batch_process_images_use_case.py   # Batch Processing
│   └── find_duplicates_use_case.py        # Duplicate Detection
└── interfaces/
    ├── __init__.py
    ├── image_processor.py          # Image Processor Interface
    ├── embedding_generator.py      # Embedding Generator Interface
    └── vector_repository_factory.py # Repository Factory
```

### Infrastructure Layer (Инфраструктурный слой)

```
infrastructure/
├── database/
│   ├── sqlite_repository.py       # SQLite Implementation
│   └── chromadb_repository.py     # ChromaDB Implementation
├── file_system/
│   └── image_file_handler.py      # File System Operations
└── external_services/
    ├── ml_model_service.py        # ML Model Service
    └── cache_service.py           # Caching Service
```

### Interfaces Layer (Слой интерфейсов)

```
interfaces/
├── cli/
│   ├── __init__.py
│   ├── commands.py                # CLI Commands
│   └── main.py                    # CLI Entry Point
├── rest_api/
│   ├── __init__.py
│   ├── routes.py                  # API Routes
│   └── server.py                  # API Server
└── web/
    ├── __init__.py
    ├── templates/                 # HTML Templates
    └── static/                    # Static Assets
```

## 🔧 Преимущества DDD архитектуры

### 1. **Чистая архитектура**
- ✅ Четкие границы между доменами
- ✅ Слабая связанность компонентов
- ✅ Высокая когезия внутри доменов

### 2. **Улучшенная тестируемость**
- ✅ Каждый домен тестируется независимо
- ✅ Легко создавать моки и стабы
- ✅ Unit тесты фокусируются на бизнес-логике

### 3. **Гибкость и расширяемость**
- ✅ Новые алгоритмы легко добавить
- ✅ Новые типы БД через репозитории
- ✅ Новые модели через конфигурацию

### 4. **Доменная экспертиза**
- ✅ Код отражает предметную область
- ✅ Терминология согласована с бизнесом
- ✅ Понятно не-техническим специалистам

### 5. **Устойчивость к изменениям**
- ✅ Изменения в инфраструктуре не затрагивают домен
- ✅ Бизнес-логика изолирована от технических деталей
- ✅ Проще мигрировать между технологиями

## 📊 Сравнение с традиционной архитектурой

| Аспект | Традиционная | DDD |
|--------|-------------|-----|
| **Структура** | Плоская | Слоистая |
| **Связанность** | Высокая | Низкая |
| **Тестируемость** | Сложная | Простая |
| **Расширяемость** | Ограниченная | Высокая |
| **Бизнес-логика** | Смешана | Изолирована |
| **Конфигурация** | Жёсткая | Гибкая |

## 🛠️ Технические детали

### Неизменяемые Value Objects

```python
@dataclass(frozen=True)
class VectorEmbedding:
    vector: np.ndarray
    model_name: str
    created_at: datetime
    metadata: Optional[Dict] = None
    
    def cosine_similarity(self, other: "VectorEmbedding") -> float:
        # Business logic encapsulated in value object
        return float(np.dot(self.vector, other.vector))
```

### Aggregate Root Pattern

```python
class VectorStore:
    """Aggregate root for vector storage operations."""
    
    def add_image(self, image: Image, embedding: VectorEmbedding) -> UUID:
        # Coordination logic for complex operations
        # Ensures consistency within the aggregate
        pass
    
    def find_similar_images(self, query: VectorEmbedding, limit: int):
        # Hide complexity of repository interactions
        pass
```

### Repository Interface

```python
class VectorRepository(Protocol):
    """Abstraction for data access."""
    
    def save(self, embedding: VectorEmbedding, image: Image) -> UUID: ...
    def find_similar(self, query: VectorEmbedding, limit: int) -> List: ...
    def find_duplicates(self, threshold: float) -> List[List]: ...
```

## 🚦 Миграция с существующей архитектуры

1. **Постепенный переход** - можно использовать старую и новую архитектуру параллельно
2. **Адаптеры** - создать адаптеры для существующих компонентов
3. **Новые функции** - новый функционал сразу в DDD архитектуре
4. **Полная миграция** - постепенный перенос существующего кода

## 📚 Дополнительные ресурсы

- [DDD Architecture Guide](DDD_ARCHITECTURE.md) - Подробное руководство
- [Examples](examples/ddd_example.py) - Практические примеры
- [API Documentation](docs/api/) - Полная документация API
- [Best Practices](docs/best_practices.md) - Рекомендации по использованию

## 🎯 Следующие шаги

1. **Реализация инфраструктуры** - создать конкретные реализации репозиториев
2. **Тестирование** - покрыть все компоненты тестами
3. **CLI интерфейс** - создать командную строку
4. **REST API** - реализовать веб-сервис
5. **Миграция** - постепенно перенести существующий код

---

**DDD архитектура** обеспечивает прочную основу для долгосрочного развития проекта, делая его более понятным, тестируемым и адаптируемым к изменениям.