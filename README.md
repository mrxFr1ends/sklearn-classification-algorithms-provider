# Sklearn Microservice

## Сборка и запуск провайдера:

```
docker build -t sklearn_provider .
docker run sklearn_provider
```

## Сборка и запуск провайдера вместе с сервером RabbitMQ:

```
docker-compose build
docker-compose up
```

## Изминение ключей и JSON структуры

В файле `config.py` в объекте `REQUEST_HANDLER_MAP` нельзя менять ключи не глубже 3 уровня!!!
Например:

```
RequestStatus.CREATE -> 'comp_response_keys' -> 'serialized_model' -> 'serializedModelData'
                   1 ->                    2 ->                  3 ->                     4
```

Ключ `serializedModelData` менять можно, все остальные нет.

Пример 1. Изминение ключа у входного параметра `features` при обучении модели

Если параметр `features` передается в `serializedModelData`, тогда конфиг будет выглядеть так:

```python
RequestStatus.TRAIN: {
    'request_handler': train_model,
    'result_status': ResponseStatus.TRAINED,
    'comp_request_keys': {
        'serialized_model': {'serializedModelData': ['model']}, 
        # Старая строчка
        # 'features': 'features', 
        # Новая строчка
        'features': {'serializedModelData': ['features']}
        'labels': 'labels'
    },
    'comp_response_keys': {
        'result_status': 'modelLabel', 
        'serialized_model': {'serializedModelData': ['model']}, 
        'metrics': 'metrics'
    },
    'topic_name': MainTopic.MAIN
},
```

Изминение ключа у выходного параметра, происходит аналогично примеру 1, образом.

Чтобы изменить ключи `modelId` и `modelLabel`, нужно их поменять в `app.py`.