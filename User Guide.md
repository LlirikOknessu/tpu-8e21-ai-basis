 # Методы ИИ

Данный документ направлен на первичную подготовку репозитория средства   
индивидуальной разработки к выполнению второй лабораторной работы.

## Ход работы


### Импорт библиотек

Импортирование пакетов происходит следующими коммандами:  

`pip install poetry` 

`poetry install` 
poetr
### Провести анализ предоставленных данных

Анализ проводится в jupyter notebook. Что это такое и как им пользоваться можно прочесть здесь:

https://practicum.yandex.ru/blog/chto-takoe-jupyter-notebook/

С помощью средств pandas и matplotlib (можно использоовать любые другие удобные вам инструменты)
провести первичный анализ данных. Он называется Exploratory Data Analysis (EDA).

Примеры проведенных анализов данных на примере других данных можно ознакомиться по ссылкам:

- https://www.kaggle.com/code/ash316/eda-to-prediction-dietanic
- https://www.kaggle.com/code/lucamassaron/eda-target-analysis
- https://www.kaggle.com/code/upadorprofzs/eda-video-game-sales

Исходя из анализа должно быть понятно:

- Какие признаки вы будете использовать для предсказания и почему?
- Как вы будете преобразовывать эти признаки и почему?
- Оценить значимость каждого признака?

В конце EDA должен быть полный и исчерпывающий вывод о том, какие у вас данные и как вы будете решать задачу.


### Реализация

Следуйте данным шагам реализации после проведения EDA:

1. импортируйте необходимые пакеты и классы;
2. составьте пайплайн подготовки данных для дальнейшего обучения моделей, основываясь на проведенном EDA;
3. предоставьте данные для работы и преобразования;
4. создайте модель регрессии и приспособьте к существующим данным;
5. проверьте результаты совмещения и удовлетворительность модели;
6. примените модель для прогнозов;
7. используйте модель многослойного персептррона аналогичным образом;
8. предоставьте все необходимые метрики и графиги обучения моделей;
9. Для построения кривых обучения и всю работу с многослойным персептроном необходимо вести, используя Tensorboard
