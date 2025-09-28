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