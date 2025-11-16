# Docker Quick Start for ChromaDB Backend

## Быстрый запуск с Docker

### Сборка образа
```bash
docker build -f Dockerfile.chromadb -t image-similarity-chromadb .
```

### Запуск контейнера
```bash
# Интерактивный режим с демонстрацией
docker run -it --rm -v $(pwd)/data:/app/data image-similarity-chromadb

# Запуск в фоне
docker run -d --name chromadb-backend -v $(pwd)/data:/app/data image-similarity-chromadb

# Подключение к запущенному контейнеру
docker exec -it chromadb-backend bash
```

### Docker Compose (рекомендуется)
```bash
# Запуск с Docker Compose
docker-compose -f docker-compose.chromadb.yml up image-similarity-chromadb

# Запуск в фоне
docker-compose -f docker-compose.chromadb.yml up -d image-similarity-chromadb

# Просмотр логов
docker-compose -f docker-compose.chromadb.yml logs -f image-similarity-chromadb

# Остановка
docker-compose -f docker-compose.chromadb.yml down
```

## Запуск примеров

### Базовый пример
```bash
# Внутри контейнера
python examples/chromadb_example.py
```

### Специфичные тесты
```bash
# Запуск тестов ChromaDB
python -m pytest tests/test_chromadb_backend.py -v

# Интерактивная оболочка
python -c "from image_similarity.chromadb_backend import ChromaDBBackend; print('ChromaDB ready!')"
```

## Сохранение данных

Контейнер автоматически монтирует директории для сохранения данных:

- `data/input` - входные изображения
- `data/output` - результаты
- `chroma_db_data` - данные ChromaDB
- `demo_images` - демонстрационные изображения

## Устранение неполадок

### Проблемы с зависимостями
```bash
# Проверка установленных пакетов
pip list | grep -E "(chromadb|sentence-transformers)"

# Переустановка зависимостей
pip install --upgrade chromadb sentence-transformers
```

### Проблемы с памятью
```bash
# Ограничение ресурсов контейнера
docker run --memory=2g --cpus=1.0 image-similarity-chromadb
```

### Проблемы с правами доступа
```bash
# Использование текущего пользователя
docker run --user $(id -u):$(id -g) -v $(pwd):/workspace image-similarity-chromadb
```

## Полезные команды

### Просмотр статистики контейнера
```bash
docker stats chromadb-backend
```

### Копирование файлов из контейнера
```bash
# Экспорт коллекции из контейнера
docker cp chromadb-backend:/app/collection_export.json ./

# Копирование демо изображений
docker cp chromadb-backend:/app/demo_images ./extracted_demo_images
```

### Мониторинг логов
```bash
# Логи в реальном времени
docker logs -f chromadb-backend

# Последние 100 строк логов
docker logs --tail 100 chromadb-backend
```

## Кастомизация

### Изменение модели эмбеддингов
```bash
# Создать кастомный образ с другой моделью
docker build -f Dockerfile.chromadb \
  --build-arg EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2 \
  -t my-chromadb .
```

### Переменные окружения
```bash
# Запуск с кастомными переменными
docker run -e EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2 \
  -e CHROMA_PERSIST_DIRECTORY=/custom/path \
  image-similarity-chromadb
```

---

*Для получения подробной информации см. [CHROMADB_INTEGRATION.md](docs/CHROMADB_INTEGRATION.md)*