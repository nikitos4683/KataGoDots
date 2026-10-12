# Задание для Claude: реализация разреженной нейросети для Точек

Этот документ — передача работы из предыдущей сессии, в которой не было доступа к компьютеру разработчика. Всё, что ниже, уже согласовано с владельцем проекта (Никита, @nikitos4683). Пересогласовывать заново не нужно. Если по ходу работы обнаружится, что решение из спецификации технически невозможно или явно вредно, сначала спроси.

## 0. С чего начать

1. Прочитай в репозитории, ветка `dots-sparse-nn-spec`:
   - `docs/DotsSparseNN.md` — **согласованная спецификация**, источник истины по признакам, головам и формату данных;
   - `docs/DotsSparseNNPlan.md` — план по этапам;
   - `docs/DotsRules.md` — правила игры и их отражение в движке;
   - `AGENTS.md` — общий гид по кодовой базе.
2. Создай рабочую ветку `dots-sparse-nn` от `dots-sparse-nn-spec` (в ней уже есть документация поверх `master-mine`). Коммить по этапам. Пушь в `origin` (`github.com/nikitos4683/KataGoDots`).
3. **Задача:** реализовать весь план (этапы 0–5 и конфиги этапа 6) под CUDA и RTX 5060 Ti, для модели `b5c384h6nbttflrs-fson-silu-rsnh`. Каждый этап сопровождается тестами, которые ты сам запускаешь.
4. **Не делай без команды владельца:**
   - не собирай «итоговый» бинарь для использования;
   - не запускай обучение и selfplay-цикл.

   Как собирать итог и как обучать, он скажет отдельно. Сборки и короткие прогоны для тестов делать можно и нужно.

## 1. Окружение на машине разработчика (Windows)

- Репозиторий: `D:\Develop\Dots\KataGoDots`. Текущая локальная ветка `master-mine`, в ней могут лежать неотслеживаемые копии `docs/Dots*.md` — те же файлы, что в ветке; перед checkout их можно удалить.
- Компилятор: MSVC из Visual Studio **18** Community. `vcvars64.bat` лежит в `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`. Для английских сообщений компилятора: `VSLANG=1033`.
- CUDA 13 через scoop: `C:/Users/I/scoop/apps/cuda/current` (nvcc там же).
- cuDNN 9.26: `D:/Develop/Dots/cudnn-windows-x86_64-9.26.0.51_cuda13-archive` (include, `lib/x64/cudnn.lib`). DLL нужны рядом с exe или в PATH.
- vcpkg (scoop): `C:/Users/I/scoop/apps/vcpkg/current`, triplet `x64-windows` (zlib, libzip).
- Генератор Ninja (из VS). Готовая рабочая сборка: `cpp/builds/cuda-ninja-release`. Её `CMakeCache.txt` — образец всех путей. Важно: там стоит `CMAKE_CUDA_ARCHITECTURES=75`, а для RTX 5060 Ti (Blackwell) нужно **120**. Новую сборку делай в отдельной папке, например `cpp/builds/cuda-sparse`.
- GPU: RTX 5060 Ti 16 GB, compute capability 12.0.
- Python с PyTorch + CUDA установлен (есть `~/.triton`). Версии проверь сам: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
- В `tmp/claude/` лежат файлы временного «исполнителя заданий» из прошлой сессии. Они больше не нужны, папку можно удалить.

## 2. Ключевые решения (кратко; подробно — в спецификации)

- **Токены ствола:**
  - `POINT` — выставленные точки вне внутренностей областей, 11 признаков;
  - `AREA` — состоявшиеся окружения: 10 скаляров + кодировщик множества вершин границы (24 признака на вершину);
  - пустые пункты токенами **не являются**.
- **Ствол** — `b5c384h6nbttflrs-fson-silu-rsnh`: 5 блоков `bottlenest2transformerropesg`, c384, mid 192, 6 голов, learnable RoPE, SwiGLU 512, `fixscaleonenorm`, SiLU, финальная spatial RMSNorm. Свёрточный стем заменяется эмбеддингами токенов.
- **RoPE** по множествам позиций: `cos_t = mean_{p∈P(t)} cos(ω·p)`, то же для sin. У точки одна позиция; у области — вершины границы (`area_rope = "mean"` по умолчанию, `"none"` для A/B).
- **Кандидаты действий** — запросы декодера с cross-attention к токенам ствола:
  - пункты из `getReasonableMoves` (8 признаков, включая `gain_my`/`gain_opp`);
  - заземление (NoPE, признак `ground_result`).

  `candidate_decoder_layers = 2` (параметр конфига), ширина 192, глобальный пулинг подмешивается в начальный вектор кандидата. На кандидата 6 policy-логитов (как у KataGo) + 2 futurepos (+8 и +32 хода), один общий softmax на M+1.
- **Ownership:** один выход на `POINT` — P(точку захватит соперник её владельца). Цель — `finalOwnership` движка.
- **Головы позиции:** все скалярные головы KataGo сохраняются (WDL, score mean/stdev, lead, vtime, td, short-term errors, scorebelief) поверх глобального пулинга. «Площадь доски» в пулинге берётся из `x_size·y_size`.
- **Глобальные признаки (8):**
  - заземление в истории;
  - коми/20;
  - заземление разумно;
  - `score_my`/20, `score_opp`/20, `(score_diff+komi)`/20;
  - `x_size`/40, `y_size`/40.
- **Данные:** C++ пишет только целочисленные координаты и факты. Всё геометрическое (`/40`, края, Фурье, RoPE) модель считает сама, одной формулой в PyTorch и C++. Массивы NPZ — §7.2 спецификации.
- **Бэкенды:** родной **CUDA** (cuDNN SDPA) и **Eigen** (CPU-эталон для тестов). ONNX не используется. Go не трогать.
- **Самоубийство** по умолчанию запрещено (`Rules::DEFAULT_DOTS`, сейчас там `multiStoneSuicideLegal = true`).

## 3. Решения реализации, принятые в прошлой сессии

Эти решения ещё не обсуждались с владельцем. Если сочтёшь нужным, допиши их в спецификацию.

1. **Токен `GLOBAL`** (один на позицию, NoPE): эмбеддинг из глобальных признаков + обучаемый вектор. Нужен, чтобы последовательность никогда не была пустой (стартовая позиция `EMPTY` даёт 0 точек, а cross-attention к пустому набору даёт NaN). Плюс, как у KataGo, `linear_global(globals)` прибавляется ко всем токенам. Последовательность: `[GLOBAL][POINT×P][AREA×A]`.
2. **Безпозиционные токены** (GLOBAL, кандидат-заземление, области при `area_rope="none"`): в learnable RoPE вращаются все пары измерений, поэтому у них q=k=0. Это принятое поведение: такой токен внимает равномерно и сам получает логит 0. Это и есть NoPE в нашей постановке.
3. **Последовательность как «доска» 1×S с маской.** Это позволяет переиспользовать блоки PyTorch (`NestedBottleneckTransformerBlock`, `NormActConv`, `RMSNormMask`) и CUDA-код ствола почти без изменений. Внутри ствола `mask_sum_hw` = число токенов. Это правильно для spatial RMSNorm и NormMask, но **не** для «площади» в gpool.
4. **RoPE через взвешенные множества позиций.** Плоский список записей `(batch, token, x, y, w)`: у точки одна запись с `w=1`, у области `|B|` записей с `w=1/|B|`. Для каждого слоя (у каждого свои `rope_freqs`, форма `(H_kv, P, 2)`) таблицы `cos/sin[b,t,h,p] = Σ w·cos/sin(ωx·x + ωy·y)`. Для кандидатов то же самое (по одной записи на пункт).
5. **Внимание в PyTorch:** подкласс `TransformerAttentionBlock`, переопределён только `forward` (таблицы cos/sin приходят через `block_shared_data`). Имена параметров не меняются, поэтому экспорт (`write_transformer_attention_block`) и C++ `TransformerAttentionDesc` переиспользуются. Блоки `NestedBottleneckTransformerBlock` создавать штатно, а их внутренние attention-модули заменить на подкласс. Cross-attention декодера — отдельный класс; формат экспорта аналогичен attention-блоку, но с отдельным тегом.
6. **Пулинг для голов** — свой, одинаковый в Python и C++:
   - среднее по токенам (сумма / число токенов);
   - смещение `sqrt(x_size·y_size) − 14` для членов `/10` и `/100 − 0.1` (как в `KataValueHeadGPool`/`KataGPool`);
   - max для пулинга декодера.

   `ValueHead.scorebelief` использует `input_global[:,-1:]` как «чётность» — подавать туда нули.
7. **Выходы в C++ раскладываются в обычный `NNOutput`**, тогда поиск не меняется:
   - логиты кандидатов → `policyProbs[NNPos::locToPos(loc)]`, заземление → индекс паса;
   - смешивание с оптимистичной policy как в Eigen `getOutput`: `p + (pOpt − p)·policyOptimism`. Экспорт пишет только выходы 0 и 5, как для v17;
   - нормализация `nneval.cpp` уже маскирует policy по `buf.dotsReasonableMoves`;
   - ownership: для `POINT` с вероятностью захвата `p` пишется `whiteOwnerMap = +p`, если владелец чёрный, и `−p`, если белый (захватывает соперник). Для точек внутри областей (не токенов) — текущее состояние (±1 захвачено / 0 своя выключенная). Для этого в постобработке `nneval.cpp` нужна отдельная ветка для разреженной модели вместо `tanh`;
   - симметрии: выходы индексированы кандидатами и токенами, а не сеткой. Симметрию применяй только к координатам на входе (4 симметрии для прямоугольного поля, 8 для квадратного); обратное преобразование выходов не нужно.

## 4. Находки по коду (чтобы не искать заново)

### Движок (`cpp/game`)

- **Биты состояния** (`dotsfield.cpp`):
  - `getActiveColor`, `getPlacedDotColor`;
  - `isTerritory` (`TERRITORY_FLAG`) — пункт внутри состоявшейся области;
  - `getEmptyTerritoryColor` — пункт внутри домика этого цвета;
  - `isGrounded` (`GROUNDED_FLAG`).
- **Области:** 4-связные компоненты пунктов с `isTerritory` и одинаковым `getActiveColor` (= владелец). Граница — активные непокрытые (`!isTerritory`) точки владельца, 4-смежные с компонентой. Это следует из состояния доски, вложенные перезахваты уже разрешены. Сверь на тестах с `Base::surrounding_locs`.
- **`movesAgo` области:** область создана k ходов назад, если в `hist.getRecentBoard(k)` не все её внутренние пункты были territory того же владельца, а в `getRecentBoard(k−1)` — все. Проверяется для k = 1…5 в пределах тех же правил истории, что в `fillRowV7Dots`: чередование цветов, пропуск `NULL_LOC` и неразумных ходов, без истории, если разумно только заземление.
- **Счёт:** `numBlackCaptures − numWhiteCaptures` = счёт белых минус чёрных (`dotsfield.cpp`, около строки 215).
- **`gain_my`/`gain_opp`:** `calculateCapturesAndTerritoriesColorsForDots` не сохраняет `captures_diff`. Считай отдельно: для пунктов, где `getOneMoveCaptureColor()` содержит нужный цвет, сделай `playMoveRecordedDots(loc, pla)` → посчитай изменение счёта → `undoDots(record)`. Это тот же приём с `const_cast`, что в `recalculateCapturesAndTerritories`. Для остальных кандидатов gain = 0.
- **`getReasonableMoves`** (`boardhistory.cpp:896`, `dotsboardhistory.cpp`):
  - уже отсекает углы, занятые пункты и ходы в чужой домик (кроме захватывающих — исключение 5.16), **независимо от флага самоубийства**;
  - `PASS_LOC` добавляется, если заземление разумно или ходов нет.
- **Финальное владение:** `calculateOwnershipAndWhiteScore` (`dotsfield.cpp:166`) — уже ровно то, что нужно для цели ownership. В `trainingwrite.cpp` это `finalOwnership`.
- **futurepos** в `trainingwrite.cpp` (около строки 815) использует `board.getColor` на досках через +8 и +32 хода. Для Точек бери `getPlacedDotColor`.

### Нейросеть C++ (`cpp/neuralnet`)

- **Версии:**
  - `modelversion.h/.cpp`: `latestModelVersionImplemented = 17`, на это есть `static_assert`. Добавь 18 только для Точек;
  - `desc.cpp` `ModelDesc(istream&)`: заголовок `version,maxX,maxY,game,gitRev,backend` → `isDotsGame`. Для v18 + Dots разбирай новую структуру (эмбеддинги, кодировщик области, ствол без `initialConv`, декодер, головы). Существующие парсеры блоков и `TransformerAttentionDesc`/`TransformerFFNDesc` переиспользуй.
- **`NNResultBuf`** (`nneval.h`) уже содержит `dotsReasonableMoves`. Добавь туда структуру разреженного входа: вектора с ёмкостью на максимум, заполняются в `fillRowBufs`.
- **`NeuralNet::getOutput(..., NNResultBuf** inputBufs, ...)`** — бэкенд читает вход прямо из `NNResultBuf`. Отличное место для упаковки батча: дополнение до максимума батча, `S` округляется вверх до кратного 64.
- **Eigen** (`eigenbackend.cpp`) уже поддерживает трансформерные блоки — CPU-эталон делай на нём.
- **CUDA** (`cudaandrocmbackend.inc`, общий с ROCm):
  - attention через cuDNN SDPA, кэш графов по ключу с `seqLen`;
  - learnable RoPE — `customCudaApplyRoPEQKLearnableRecompute`: позиция выводится из индекса через `nnXLen`. Нужно новое ядро, применяющее поворот по готовым таблицам cos/sin, плюс ядро построения таблиц из записей `(token, x, y, w)` и частот слоя;
  - нужны segment-reduce (mean/max) для кодировщика области, cross-attention (SDPA с разной длиной Q и KV) и gather/scatter выходов.
- Существующие тест-модели лежат в `cpp/tests/models`. Пример вывода рабочей CUDA-сборки — `cpp/builds/cuda-ninja-release/transformer-smoke.log`.

### Python (`python/`)

- **`katago/train/model_pytorch.py`** (4400 строк):
  - `TransformerAttentionBlock` (около строки 2106; поддерживает позиции по токенам только для inline registers);
  - `NestedBottleneckTransformerBlock` (около 1946);
  - `NormMask` (fixscaleonenorm: внутренние нормы аффинные, BN только `is_last_batchnorm`; при `-rsnh` финальная норма — `RMSNormMask` spatial, так что BN нет вовсе);
  - `ValueHead` (около 2838), `Model.initialize` (около 3678): fixscale `1/sqrt(i+1)` по блокам.
- **`modelconfigs.py`:** конфиг `b5c384h6nbttflrs` (около строки 1223), суффиксы `-fson`/`-silu`/`-rsnh` (около 2030–2125). Нужен разреженный вариант, например флаг `"sparse_dots": True` + `"area_rope"`, `"candidate_decoder_layers"`, `"candidate_decoder_channels"`, и имя `b5c384h6nbttflrs-fson-silu-rsnh` должно давать разреженную модель при `-game dots` (или отдельное имя с суффиксом — согласуй с владельцем).
- **`export_model_pytorch.py`:** формат `.bin` (заголовок + текстовые теги + `@BIN@` little-endian float). Для v18-Dots нужна отдельная функция записи модели.
- **`train.py`:** точки интеграции:
  - создание `Model(...)` (строки ~926, ~977);
  - `data_processing_pytorch.read_npz_training_data` (~1687, ~1966);
  - вызов модели `ddp_model(batch["binaryInputNCHW"], batch["globalInputNC"], ...)`;
  - `raw_model.postprocess_output`;
  - `Metrics(world_size, raw_model)` и `metrics_dict_batchwise`;
  - `add_reg_dict` — группы регуляризации.

  Для разреженной модели нужен загрузчик, вызов с batch-словарём и свой класс метрик (наследник `Metrics`). `torch.compile` с переменной длиной последовательности: начни с `dynamic=True` или отключения компиляции для разреженной модели и запиши это.
- **`shuffle.py`:** новые массивы и повторное дополнение при склейке.
- **Тесты:** `pytest` из корня (`pytest.ini` → `python/tests`).

## 5. Порядок работы и критерии готовности

Следуй `docs/DotsSparseNNPlan.md`. Коротко:

| Этап | Готово, когда |
|---|---|
| 0 | Ветка, `DEFAULT_DOTS` без самоубийства, v18 объявлена; `katago runtests` и `pytest` зелёные (обнови ожидаемые выводы тестов, если они зависели от флага самоубийства, и объясни это в коммите) |
| 1 | Извлечение `DotsSparseInput` + `katago dumpsparse`; ручные позиции и стресс-тест: сумма `insideOpp` по моим областям = мои захваты; перезахват ровно одной области даёт `captures_diff = −(n_X+n_O+|B|)`; `grounded` одинаков на границе; эквивариантность симметриям; кандидаты = `getReasonableMoves` |
| 2 | PyTorch-модель, потери, загрузчик, экспорт; тесты: инвариантность к паддингу, эквивариантность, тождество усреднённого RoPE, переобучение крошечного набора, круговой экспорт |
| 3 | v18 desc + Eigen; выходы Eigen = PyTorch на наборе позиций (FP32, ~1e-4); `katago gtp` со случайной моделью ходит |
| 4 | `trainingwrite`, `shuffle.py`, загрузчик реальных файлов; короткий сквозной прогон (несколько партий selfplay случайной сетью → shuffle → несколько шагов train → export → загрузка в C++) как тест |
| 5 | CUDA: совпадение с Eigen (FP32 и FP16 с разумными допусками), `katago benchmark` работает; `CMAKE_CUDA_ARCHITECTURES=120` |
| 6 | Конфиги `cpp/configs/*dots*` и `training/` для новой модели (без запуска обучения), обновлённые `docs/` и `AGENTS.md` |

По завершении каждого этапа: коммит + push + короткий отчёт владельцу (что сделано, какие тесты прошли, что осталось). Сообщения ему пиши по-русски.
