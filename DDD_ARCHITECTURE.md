# Domain Driven Design (DDD) Architecture

## Обзор

Проект Image Similarity Toolkit был переработан с использованием принципов **Domain Driven Design (DDD)** для создания более чистой, модульной и поддерживаемой архитектуры.

## Структура DDD

### Domain Layer (Ядро домена)
```
src_ddd/domain/
├── image_processing/          # Домен обработки изображений
├── vector_storage/            # Домен хранения векторов
├── similarity_search/         # Домен поиска похожести
├── database_management/       # Домен управления БД
└── configuration/             # Домен конфигурации
```

### Application Layer (Слой приложения)
```
src_ddd/application/
├── use_cases/                 # Сценарии использования
└── interfaces/                # Интерфейсы приложения
```

### Infrastructure Layer (Инфраструктурный слой)
```
src_ddd/infrastructure/
├── database/                  # Реализации БД
├── file_system/              # Файловая система
└── external_services/         # Внешние сервисы
```

### Interfaces Layer (Слой интерфейсов)
```
src_ddd/interfaces/
├── cli/                       # Командная строка
├── rest_api/                  # REST API
└── web/                       # Веб интерфейс
```

## Ключевые принципы DDD

### 1. Ubiquitous Language (Единый язык)
Все классы и методы используют терминологию из предметной области:
- `Image` - изображение как value object
- `VectorEmbedding` - векторное представление
- `SimilarityQuery` - запрос на поиск похожести
- `VectorStore` - хранилище векторов (aggregate root)

### 2. Bounded Contexts (Ограниченные контексты)
Проект разделён на логические домены:

#### Image Processing Domain
- **Image** (Value Object) - неизменяемое представление изображения
- **ImageProcessor** (Service) - обработка изображений
- **EmbeddingExtractor** (Service) - извлечение эмбеддингов

#### Vector Storage Domain
- **VectorEmbedding** (Value Object) - неизменяемый вектор
- **VectorRepository** (Interface) - абстракция доступа к данным
- **VectorStore** (Aggregate Root) - основной объект управления векторами

#### Similarity Search Domain
- **SimilarityQuery** (Value Object) - запрос поиска
- **SimilarityResult** (Value Object) - результат поиска
- **SimilarityCalculator** (Service) - расчёт похожести
- **DuplicateDetector** (Service) - обнаружение дубликатов

### 3. Value Objects (Объекты-значения)
Все value objects неизменяемы и определяются своими атрибутами:

```python
@dataclass(frozen=True)
class Image:
    path: Path
    dimensions: tuple[int, int]
    format: str
    file_hash: str
    metadata: Optional[dict] = None
```

### 4. Aggregates (Агрегаты)
**VectorStore** выступает как aggregate root, который:
- Координирует операции с векторами
- Управляет кэшированием
- Обеспечивает транзакционность
- Скрывает детали реализации репозиториев

### 5. Repository Pattern
Абстракция доступа к данным:

```python
class VectorRepository(Protocol):
    def save(self, embedding: VectorEmbedding, image: Image) -> UUID: ...
    def find_similar(self, query: VectorEmbedding, limit: int) -> List: ...
    def find_duplicates(self, threshold: float) -> List[List]: ...
```

### 6. Domain Services
Сервисы, содержащие бизнес-логику, которая не принадлежит конкретным объектам:

- **SimilarityCalculator** - расчёт похожести между векторами
- **DuplicateDetector** - обнаружение дубликатов
- **VectorRepository** - операции с хранилищем

## Преимущества DDD архитектуры

### 1. Чистая архитектура
- **Ясные границы** между доменами
- **Слабая связанность** между компонентами
- **Высокая когезия** внутри доменов

### 2. Улучшенная тестируемость
- Каждый домен может тестироваться независимо
- Моки и стабы легко создавать для интерфейсов
- Unit тесты фокусируются на бизнес-логике

### 3. Гибкость и расширяемость
- Новые алгоритмы похожести легко добавить
- Новые типы БД через репозитории
- Новые модели через конфигурацию

### 4. Доменная экспертиза
- Код отражает предметную область
- Терминология согласована с бизнес-требованиями
- Проще для понимания не-техническими специалистами

### 5. Устойчивость к изменениям
- Изменения в инфраструктуре не затрагивают домен
- Бизнес-логика изолирована от технических деталей
- Проще мигрировать между технологиями

## Пример использования

```python
from src_ddd.domain import (
    Image, VectorEmbedding, SimilarityQuery, VectorStore
)
from src_ddd.infrastructure import SQLiteVectorRepository

# Создание инфраструктуры
repository = SQLiteVectorRepository(db_path="embeddings.db")
vector_store = VectorStore(repository)

# Работа с доменными объектами
image = Image.from_file(Path("image.jpg"))
embedding = VectorEmbedding(vector=np.array([...]), model_name="resnet")

# Использование aggregate root
embedding_id = vector_store.add_image(image, embedding)

# Поиск похожих
query = SimilarityQuery.from_embedding(embedding)
similar_results = vector_store.find_similar_images(query, limit=10)
```

## Сравнение с предыдущей архитектурой

| Аспект | Предыдущая | DDD |
|--------|------------|-----|
| **Структура** | Плоская, все в одном модуле | Слоистая, по доменам |
| **Связанность** | Высокая | Низкая |
| **Тестируемость** | Сложная | Простая |
| **Расширяемость** | Ограниченная | Высокая |
| **Бизнес-логика** | Смешана с технической | Изолирована |
| **Конфигурация** | Жёстко заданная | Гибкая, через объекты |

## Следующие шаги

1. **Реализация инфраструктуры** - создать конкретные реализации репозиториев
2. **Создание интерфейсов** - CLI, REST API, веб-интерфейс
3. **Тестирование** - покрыть все доменные объекты и use cases
4. **Миграция** - постепенно перенести существующий код
5. **Документация** - подробные примеры использования

## Заключение

DDD архитектура обеспечивает:
- **Лучшую организацию кода**
- **Повышенную поддерживаемость**
- **Упрощённое тестирование**
- **Большую гибкость для будущих изменений**

Это создаёт прочную основу для долгосрочного развития проекта.