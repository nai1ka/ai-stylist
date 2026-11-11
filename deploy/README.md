# Развёртывание https://nai1ka.ru/stylist/

Предполагается Linux x64 сервер с Docker Compose и существующим Nginx на хосте, уже обслуживающим nai1ka.ru по HTTPS. Если Nginx работает в контейнере, `127.0.0.1:8501` внутри него не указывает на хост: эта схема требует адаптации сети.

## На Mac: собрать и упаковать

Из корня проекта:

```bash
bash scripts/package_deployment.sh
```

Скрипт собирает linux/amd64 CPU-образ, проверяет импорты и строгую загрузку checkpoint внутри контейнера, сохраняет образ и четыре локальные модели в `release/stylist/`. Сборка требует интернета; токены в архив не включаются. Датасет и виртуальное окружение не переносятся. Сборка и запуск в среде агента не проверены из-за запрета доступа к Docker socket.

Перенос (замените USER и SERVER_IP на SSH-логин и адрес сервера):

```bash
scp -r release/stylist USER@SERVER_IP:~/
ssh USER@SERVER_IP
```

## На сервере

Команды Docker предполагают, что текущий пользователь имеет доступ к Docker (иначе нужен sudo).

```bash
cd ~/stylist
docker load -i ai-stylist-amd64.tar.gz
tar -xzf models.tar.gz
docker compose up -d
docker compose ps
curl -f http://127.0.0.1:8501/stylist/_stcore/health
```

При распаковке под другим пользователем убедитесь, что UID 10001 внутри контейнера может читать файлы моделей и проходить по их каталогам.

## Существующий Nginx

Сохраните фрагмент отдельно:

```bash
sudo mkdir -p /etc/nginx/snippets
sudo cp nginx-stylist.conf /etc/nginx/snippets/stylist.conf
```

В существующий **HTTPS-блок** `server { ... }` с `server_name nai1ka.ru` добавьте:

```nginx
include /etc/nginx/snippets/stylist.conf;
```

Затем проверьте и примените:

```bash
sudo nginx -t
sudo systemctl reload nginx
curl -I https://nai1ka.ru/stylist/
```

Открыть **https://nai1ka.ru/stylist/**. `/stylist` перенаправляется на `/stylist/`. Префикс сохраняется при proxy_pass, а Streamlit запускается с `server.baseUrlPath=stylist`. WebSocket включён для интерактивного интерфейса. Остальные пути домена продолжают обслуживаться существующей конфигурацией. Существующий сертификат nai1ka.ru действует и для этого пути; отдельный сертификат для пути не нужен.

Порт 8501 доступен только через loopback сервера. Учётной записи в самом AI-Stylist нет: путь доступен посетителям домена, если ограничения доступа не настроены в Nginx. Модель предназначена для научного/исследовательского использования согласно её лицензии.

## Проверка и обновление

Проверьте страницу, загрузку фото и все три шага анализа, отображение подготовленного входа, ручной выбор и смену фото. Healthcheck подтверждает ответ сервера Streamlit, а не полный анализ фотографии.

```bash
docker compose logs --tail=100 -f
# Остановка:
docker compose down
```

Для обновления пересоберите и перенесите новый архив образа, выполните `docker load`, затем `docker compose up -d --force-recreate`. Если веса не изменились, повторно переносить и распаковывать models.tar.gz не требуется. Не используйте docker import для образа — нужен docker load.

Размер CPU-образа предварительно оценивается в 2–4 ГБ плюс около 1,1 ГБ весов. Архивы сжаты, но при загрузке и распаковке нужны дополнительные дисковые ресурсы; точные размеры покажут `du -sh` и `docker image ls`. Архивы после успешного развёртывания можно убрать вручную.

Документация: https://docs.docker.com/reference/cli/docker/image/save/ ; https://nginx.org/en/docs/http/websocket.html ; https://docs.streamlit.io/develop/api-reference/configuration/config.toml .
