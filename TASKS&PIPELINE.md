# Plan for 6 weeks
## 1. Research & architecture
- watch videos
- highlight main topics
- research topics
- Model architecture: data, models, learning, inference
- specify restrictions on developing.
- fomulate testing, metrics
- Model main classes and structures on paper
- Create and test git
- Split responsibilities
**Result:** README.md with all pipeline structures

## 2. Data & Bigram
- Preparing data
- Create tokenizer
- Create bigram
- Test bigram
**Result:** working BiGram

## 3. MLP
- Create layers
- Break down backpropagation, forward pass
- Create MLP
- Test MLP

## 4. Enhancing learning
- Add train/val, batching, learning rate, logging
- Cleaning code

## 5. Mini-GPT
- Create self-attention
- Causal mask
- Build transformer

## 6. Finalizing
- Re-write README.md
- Polish up the code
- Vibe code frontend


# Topics to cover
- **Линейная алгебра: векторы, матрицы, размерности** — все данные, веса и активации модели представлены в виде матриц; без этого невозможно собрать ни один слой  
- **Скалярное произведение** — основа линейного слоя и вычисления логитов  
- **Условная вероятность** — формулировка задачи language model как P(next_token | context)  
- **Языковая модель (language model)** — общее понимание, что именно вы реализуете и обучаете  
- **Unigram модель** — базовый бенчмарк и стартовая точка перед нейросетью  
- **Bigram модель** — первая модель с контекстом, логически переходит в neural bigram  
- **Trigram модель** — демонстрация комбинаторного взрыва и причины перехода к нейросетям  
- **Char-level токенизация** — представление текста в виде последовательности токенов для модели  
- **Logits** — выход линейного слоя перед softmax, с ними считается loss  
- **Softmax** — преобразование логитов в распределение вероятностей по символам  
- **Cross-entropy loss** — функция ошибки, по которой обучается модель  
- **Embedding слой** — преобразование token id в обучаемый вектор признаков  
- **Линейный (fully connected) слой** — основной обучаемый слой модели  
- **ReLU** — добавление нелинейности между линейными слоями  
- **Градиент** — величина, определяющая, как менять веса при обучении  
- **Chain rule** — механизм распространения градиентов назад по слоям  
- **Backpropagation** — алгоритм вычисления градиентов всех параметров модели  
- **Stochastic Gradient Descent (SGD)** — обновление весов модели на основе градиентов  
- **Вычислительный граф** — структура зависимостей операций для реализации backward  
- **Autograd (минимальная реализация)** — автоматизация backprop для всех операций  
- **Training loop** — цикл обучения: forward → loss → backward → update  
- **Inference и text generation** — использование обученной модели для генерации текста  
- **Temperature sampling** — контроль случайности при генерации символов  
- **C++ реализация тензоров** — хранение данных и выполнение матричных операций  
- **CMake и структура ML-проекта** — сборка проекта и связка всех модулей в единое приложение  
