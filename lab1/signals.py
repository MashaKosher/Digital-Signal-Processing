import math
import random
from typing import List


def _clamp(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def _validate_common_params(
    duration: float, rate: int, amplitude: float
) -> tuple[int, float]:
    if duration <= 0:
        raise ValueError("duration_s должно быть > 0")
    if rate <= 0:
        raise ValueError("sample_rate_hz должно быть > 0")
    num_samples = int(round(duration * rate)) # кол-во точек
    safe_amplitude = _clamp(amplitude, 0.0, 1.0)
    return num_samples, safe_amplitude


def generate_sine(
    freq: float, # hz
    duration: float, # seconds
    rate: int = 44100, # hz
    amplitude: float = 1.0,
) -> List[float]:
    """Генерация синусоидального сигнала.

    Возвращает список значений в диапазоне [-amplitude, amplitude].
    """
    num_samples, amp = _validate_common_params(duration, rate, amplitude)
    if freq < 0:
        raise ValueError("frequency_hz должно быть >= 0")

    # print("Num samples: ", num_samples)

    two_pi_f = 2.0 * math.pi * freq
    # print("two pi f: ", two_pi_f)

    samples: List[float] = []
    for n in range(num_samples):
        t = n / rate
        samples.append(amp * math.sin(two_pi_f * t))
        # print("t: ", t, " on step: ", n)

    # print("Samples: ", samples)
    return samples


def generate_triangle(
    freq: float,
    duration: float,
    rate: int = 44100,
    amplitude: float = 1.0,
) -> List[float]:
    """Генерация треугольного сигнала."""
    num_samples, amp = _validate_common_params(duration, rate, amplitude)
    if freq < 0:
        raise ValueError("frequency_hz должно быть >= 0")

    samples: List[float] = []
    for n in range(num_samples):
        phase = (freq * (n / rate)) % 1.0  # [0,1)
        # Треугольник в [-1, 1]: 2*abs(2*phase - 1) - 1
        tri = 2.0 * abs(2.0 * phase - 1.0) - 1.0
        samples.append(amp * tri)
    return samples


def generate_sawtooth(
    freq: float,
    duration: float,
    rate: int = 44100,
    amplitude: float = 1.0,
) -> List[float]:
    """Генерация пилообразного сигнала (линейный спад -> скачок)."""
    num_samples, amp = _validate_common_params(duration, rate, amplitude)
    # if freq < 0:
    #     raise ValueError("frequency_hz должно быть >= 0")
        
    samples: List[float] = []
    for n in range(num_samples):
        phase = (freq * (n / rate)) % 1.0  # [0,1)
        saw = 2.0 * phase - 1.0
        samples.append(amp * saw)
    return samples

def generate_noise(
    duration: float,
    rate: int = 44100,
    amplitude: float = 1.0,
) -> List[float]:
    """Генерация белоподобного шума (равномерное распределение)."""
    num_samples, amp = _validate_common_params(duration, rate, amplitude)
    samples: List[float] = []
    for _ in range(num_samples):
        # samples.append(amp * random.uniform(-1.0, 1.0))
        samples.append(amp * random.gauss(0, 1))
    return samples

def generate_pulse(
    freq: float,
    duration: float,
    duty_cycle: float = 0.5,
    rate: int = 44100,
    amplitude: float = 1.0,
) -> List[float]:
    """Генерация прямоугольного импульса с заданной скважностью.

    duty_cycle — доля периода в диапазоне [0, 1], когда сигнал положительный.
    Скважность D обычно определяют как T/Tположительный, но здесь используем duty_cycle.
    """
    num_samples, amp = _validate_common_params(duration, rate, amplitude)
    if freq < 0:
        raise ValueError("frequency_hz должно быть >= 0")

    dc = _clamp(duty_cycle, 0.0, 1.0)
    
    samples: List[float] = []
    for n in range(num_samples):
        phase = (freq * (n / rate)) % 1.0
        value = amp if phase < dc else -amp
        samples.append(value)
    return samples


# ----------------------- Сложение сигналов  -----------------------

def mix_signals(signals_list: List[List[float]], normalize: bool = True) -> List[float]:
    """Суммирует несколько монофонических сигналов в один.

    - Выравнивание по минимальной длине.
    - При normalize=True выполняется нормализация, если пик > 1.0.
    """
    if not signals_list:
        return []

    min_len = min(len(s) for s in signals_list)
    mixed: List[float] = [0.0] * min_len

    for s in signals_list:
        for i in range(min_len):
            mixed[i] += s[i]

    if normalize and mixed:
        peak = max(abs(x) for x in mixed)
        if peak > 1.0:
            scale = 1.0 / peak
            mixed = [x * scale for x in mixed]

    return mixed


# ----------------------- Модуляция -----------------------

def _generate_modulator(
    mod_type: str,
    mod_freq: float,
    duration: float,
    rate: int,
    duty: float = 0.5,
) -> List[float]:
    """Генерирует модулирующий сигнал в диапазоне [-1, 1]."""
    mod_type = mod_type.lower()
    if mod_type == "sine":
        return generate_sine(mod_freq, duration, rate, 1.0)
    if mod_type == "triangle":
        return generate_triangle(mod_freq, duration, rate, 1.0)
    if mod_type == "saw":
        return generate_sawtooth(mod_freq, duration, rate, 1.0)
    if mod_type == "pulse":
        return generate_pulse(mod_freq, duration, duty, rate, 1.0)
    raise ValueError(f"Неизвестный тип модуляции: {mod_type}")


def _wave_value_from_phase(
    wave_type: str,
    phase_cycles: float,
    amplitude: float,
    duty: float = 0.5,
) -> float:
    """Возвращает значение волны по нормализованной фазе в циклах [0,1)."""
    p = phase_cycles % 1.0
    wt = wave_type.lower()
    if wt == "sine":
        return amplitude * math.sin(2.0 * math.pi * p)
    if wt == "triangle":
        return amplitude * (2.0 * abs(2.0 * p - 1.0) - 1.0)
    if wt == "saw":
        return amplitude * (2.0 * p - 1.0)
    if wt == "pulse":
        return amplitude if p < duty else -amplitude
    raise ValueError(f"Неизвестный тип волны: {wave_type}")


def apply_amplitude_modulation(
    carrier_samples: List[float],
    modulator_samples: List[float],
    depth: float = 1.0,
) -> List[float]:
    """AM: y[n] = (1 + m * mod[n]) * carrier[n], где mod в [-1,1], m∈[0,1]."""
    if not carrier_samples:
        return []
    m = _clamp(depth, 0.0, 1.0)
    length = min(len(carrier_samples), len(modulator_samples))
    out: List[float] = [0.0] * length
    for i in range(length):
        out[i] = (1.0 + m * modulator_samples[i]) * carrier_samples[i]
    return out

def generate_fm(
    wave_type: str,
    carrier_freq: float,
    duration: float,
    rate: int,
    amplitude: float,
    mod_type: str,
    mod_freq: float,
    deviation_hz: float,
    carrier_duty: float = 0.5,
    mod_duty: float = 0.5,
) -> List[float]:
    """FM: f_i[n] = f_c + Δf * mod[n], mod в [-1,1].

    Вычисляет фазу накоплением нормализованных приращений и строит выборки по фазе.
    """
    if carrier_freq < 0 or deviation_hz < 0 or mod_freq < 0:
        raise ValueError("Частоты и девиация должны быть >= 0")
    num_samples, amp = _validate_common_params(duration, rate, amplitude)
    mod = _generate_modulator(mod_type, mod_freq, duration, rate, mod_duty)
    length = min(num_samples, len(mod))
    samples: List[float] = [0.0] * length
    phase_cyc = 0.0  # фаза в циклах (а не в радианах)
    inv_rate = 1.0 / rate
    for n in range(length):
        inst_freq = carrier_freq + deviation_hz * mod[n]
        if inst_freq < 0:
            inst_freq = 0.0
        phase_cyc += inst_freq * inv_rate  # Δфаза в циклах
        samples[n] = _wave_value_from_phase(wave_type, phase_cyc, amp, carrier_duty)
    return samples


# ----------------------- Эффекты -----------------------

def apply_echo(
    samples: List[float],
    delay_seconds: float,
    decay: float = 0.5,
    rate: int = 44100,
    num_echoes: int = 3,
) -> List[float]:
    """Добавляет эффект эха к сигналу.

    Args:
        samples: входной сигнал
        delay_seconds: задержка между эхом в секундах
        decay: коэффициент затухания эха (0.0-1.0)
        rate: частота дискретизации
        num_echoes: количество повторяющихся эхо

    Returns:
        сигнал с добавленным эхом
    """
    if not samples:
        return []

    if delay_seconds <= 0:
        raise ValueError("delay_seconds должно быть > 0")
    if not (0.0 <= decay <= 1.0):
        raise ValueError("decay должен быть в диапазоне [0.0, 1.0]")
    if num_echoes < 1:
        raise ValueError("num_echoes должно быть >= 1")

    # Вычисляем задержку в сэмплах
    delay_samples = int(round(delay_seconds * rate))
    if delay_samples >= len(samples):
        # Если задержка больше длины сигнала, просто возвращаем оригинал
        return samples.copy()

    # Создаем копию оригинального сигнала
    result = samples.copy()

    # Добавляем эхо
    current_decay = decay
    for echo_num in range(num_echoes):
        start_idx = delay_samples * (echo_num + 1)

        # Проверяем, что не выходим за границы
        if start_idx >= len(result):
            break

        # Добавляем эхо к результату
        for i in range(start_idx, len(result)):
            original_idx = i - start_idx
            if original_idx < len(samples):
                result[i] += current_decay * samples[original_idx]

        current_decay *= decay  # каждое следующее эхо слабее

    # Нормализуем результат, чтобы избежать клиппинга
    if result:
        peak = max(abs(x) for x in result)
        if peak > 1.0:
            scale = 1.0 / peak
            result = [x * scale for x in result]

    return result