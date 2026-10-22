# RVC Studio

RVC Studio — desktop-приложение для realtime-конвертации голоса на базе RVC и ONNX. Интерфейс позволяет импортировать RVC-модель из ZIP, выбрать аудиоустройства и настроить сдвиг высоты тона.

> Проект в разработке. Для работы нужны совместимая RVC ONNX-модель и базовые ONNX-модели ContentVec/RMVPE.

## Быстрый запуск

### Windows

1. Установите [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).
2. Откройте PowerShell в папке проекта и выполните:

   ```powershell
   .\scripts\setup.ps1
   .\scripts\run.ps1
   ```

Либо запустите `run.bat` двойным щелчком (после установки .NET SDK).

### Linux

1. Установите [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0), PortAudio и `curl`.
2. Выполните из корня проекта:

   ```bash
   ./scripts/setup.sh
   ./scripts/run.sh
   ```

Пакеты PortAudio: Debian/Ubuntu — `sudo apt install libportaudio2`; Fedora — `sudo dnf install portaudio`.

Установщик скачивает **обе необходимые базовые модели — `contentvec.onnx` и `rmvpe.onnx`** — в `RVCStudio/assets/models/` и проверяет их SHA-256. Размер загрузки около 707 MiB. Установить модели отдельно, не запуская настройку приложения:

```bash
# Linux
./scripts/install-models.sh
```

```powershell
# Windows PowerShell
.\scripts\install-models.ps1
```

Установка моделей также выполняется автоматически командами `scripts/setup.sh` и `scripts/setup.ps1`.

## Сборка и публикация

```bash
dotnet restore RVCStudio.sln
dotnet build RVCStudio.sln -c Release
```

Публикация автономной папки для Linux x64:

```bash
dotnet publish RVCStudio/RVCStudio.csproj -c Release -r linux-x64 --self-contained true -o publish/linux-x64
```

Для Windows замените RID на `win-x64` (или `win-arm64`). Для GPU-ускорения требуется совместимая CUDA-среда.

## Добавление голосовой модели

Перетащите ZIP-архив модели в окно приложения или нажмите «Выбрать ZIP-файл». Архив должен содержать совместимую RVC generator-модель ONNX. Модели пользователя хранятся в `Models` рядом с приложением.

Подробности о форматах и базовых моделях: [`RVCStudio/Models/README.md`](RVCStudio/Models/README.md). Голосовую модель пользователь предоставляет самостоятельно.

## Структура проекта

- `RVCStudio/` — приложение Avalonia, аудиодвижок и UI.
- `RVCStudio/converter/` — исходный код и сборка конвертера RVC `.pth` → ONNX.
- `RVCStudio/assets/models/` — общие модели ContentVec/RMVPE и notices.
- `download-models.sh`, `download-models.ps1` — загрузка базовых моделей.
- `scripts/` — настройка, запуск и публикация.

Откройте `RVCStudio.sln` в Rider/Visual Studio или запустите `dotnet run --project RVCStudio/RVCStudio.csproj`.

## Лицензии

Проверяйте лицензии зависимостей и сторонних моделей перед распространением сборок. Лицензия и происхождение ContentVec/RMVPE приведены в `RVCStudio/assets/models/THIRD_PARTY_NOTICES.md` и README моделей. Пользовательские RVC-модели могут иметь отдельные условия распространения.
