Синус
python3 -m main sine -f 440 -t 2 -r 44100 -a 0.8 -o sine_440.wav

Прямоугольный
python3 -m main pulse -f 1000 -d 25 -t 2 -o pulse_1kHz_25.wav

Треугольный 
python3 -m main triangle -f 500 -t 2 -o triangle_500.wav

Пилообразный
python3 -m main saw -f 300 -t 2 -o saw_300.wav

Шум
python3 -m main noise -t 3 -o noise.wav

# Полифония 
3 синусоиды
python3 -m main sine --freqs 440,550,660 -t 3 -r 44100 -a 0.6 -o chord.wav

3 синусоиды + 2 пилообразных + 1 шум
python3 -m main --poly "sine:440,550,660;saw:330,495;noise" -t 3 -r 44100 -a

Отключить нормализацию при смешивании (возможен клиппинг):
python3 -m main sine --freqs 440,550,660 -t 3 --no-normalize -o chord_raw.wav

# Модуляция
### AM (амплитудная)
Моно (синус 440 Гц), модулятор синус 5 Гц, глубина 0.8:
python3 -m main sine -f 440 -t 3 -r 44100 -a 0.8 --am sine:5:0.8 -o am_sine.wav

Поли (после сложения 3 синусов и 2 пил, применить AM):
python3 -m main --poly "sine:440,550,660;saw:330,495" -t 3 -r 44100 -a 0.6 --am sine:5:0.8 -o mix_am.wav

### FM (частотная)
python3 -m main sine -f 440 -t 3 -r 44100 -a 0.8 --fm triangle:3:50 -o fm_sine.wav

Моно (синус 440 Гц), модулятор треугольник 3 Гц, девиация 50 Гц:
python3 -m main saw -f 330 -t 3 --fm sine:2:80 -o fm_saw.wav

### AM и FM одновременно (сначала FM, потом AM)
Моно: FM (triangle:3:50) → AM (sine:5:0.8):
python3 -m main sine -f 440 -t 3 --fm triangle:3:50 --am sine:5:0.8 -o fm_am_sine.wav

Поли: сложение 3 синусов и 2 пил → AM:
python3 -m main --poly "sine:440,550,660;saw:330,495" -t 3 --am sine:5:0.8 -o poly_am.wav

# Эффекты
### Эхо
Простое эхо: задержка 0.3с, затухание 0.4, 2 эха
python3 -m main sine -f 440 -t 2 --echo 0.3:0.4:2 -o sine_echo.wav

Сильное эхо: задержка 0.5с, слабое затухание 0.3, 4 эха
python3 -m main saw -f 300 -t 1.5 --echo 0.5:0.3:4 -o saw_echo.wav

Эхо на аккорде
python3 -m main --poly "sine:440,550,660" -t 3 --echo 0.2:0.6:3 -o chord_echo.wav

Эхо после модуляции
python3 -m main sine -f 440 -t 4 --am triangle:2:0.8 --echo 0.4:0.5:2 -o am_echo.wav

Шум с быстрым эхом
python3 -m main noise -t 2 --echo 0.1:0.8:5 -o noise_echo.wav

# Сирены экстренных служб
### Скорая помощь (медленный "вой" 800-1200 Гц)
python3 -m main sine -f 900 -t 4 --fm triangle:1.5:300 --echo 0.05:0.4:3 -o ambulance_siren.wav

### Полиция (быстрый "вой" 600-1000 Гц)
python3 -m main sine -f 700 -t 3 --fm sine:2:250 --echo 0.08:0.3:2 -o police_siren.wav

### Пожарная машина (громкий "рёв" 500-1100 Гц)
python3 -m main sine -f 600 -t 2.5 --fm triangle:3:400 --echo 0.03:0.5:4 -o fire_truck_siren.wav

### Hi-Lo сирена (двухтональная, 800-1000 Гц)
python3 -m main sine --poly "sine:800,1000" -t 3 --fm triangle:1:100 --echo 0.1:0.3:2 -o hi_lo_ambulance.wav