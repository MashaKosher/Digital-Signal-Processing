import argparse
import os

from my_io import write_wav_mono
import signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация WAV сигналов: синус, импульс, треугольный, пила, шум.",
    )
    parser.add_argument(
        "type",
        nargs="?",
        choices=["sine", "pulse", "triangle", "saw", "noise"],
        help="Тип сигнала (не обязателен при использовании --poly)",
    )
    parser.add_argument("-f", "--freq", type=float, default=440.0, help="Частота, Гц")
    parser.add_argument(
        "--freqs",
        type=str,
        default=None,
        help="Список частот через запятую для полифонии (например 440,550,660)",
    )
    parser.add_argument(
        "--poly",
        type=str,
        default=None,
        help=(
            "Смешивание разных типов: формат type:freqs;type:freqs. "
            "Пример: sine:440,550,660;saw:330,495. Для noise частоты не требуются"
        ),
    )
    # Модуляция
    parser.add_argument(
        "--am",
        type=str,
        default=None,
        help=(
            "Амплитудная модуляция: формат mod_type:mod_freq:depth. "
            "Пример: --am sine:5:0.8"
        ),
    )
    parser.add_argument(
        "--fm",
        type=str,
        default=None,
        help=(
            "Частотная модуляция: формат mod_type:mod_freq:deviation_hz. "
            "Пример: --fm triangle:3:50"
        ),
    )
    parser.add_argument(
        "-d",
        "--duty",
        type=float,
        default=50.0,
        help="Скважность для pulse, в процентах или долях (0..1)",
    )
    parser.add_argument(
        "-t", "--duration", type=float, default=2.0, help="Длительность, секунды"
    )
    parser.add_argument(
        "-r", "--rate", type=int, default=44100, help="Частота дискретизации, Гц"
    )
    parser.add_argument(
        "-a", "--amp", type=float, default=0.8, help="Амплитуда (0..1)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Отключить нормализацию при смешивании",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        default=None,
        help="Путь к выходному WAV-файлу",
    )
    # Эффекты
    parser.add_argument(
        "--echo",
        type=str,
        default=None,
        help=(
            "Добавить эффект эха: формат delay:decay:echoes. "
            "Пример: --echo 0.3:0.5:2"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sig_type = args.type
    freq: float = args.freq
    freqs_arg = args.freqs
    duty_in = args.duty
    duration: float = args.duration
    rate: int = args.rate
    amp: float = args.amp

    # Интерпретация duty: если > 1, считаем процентами
    duty = duty_in / 100.0 if duty_in > 1.0 else duty_in

    # Вычислим имя файла по умолчанию, если не задано
    if args.outfile:
        outfile = args.outfile
    else:
        if args.poly:
            base = "poly"
        else:
            base = sig_type
            if sig_type != "noise":
                base += f"_{int(round(freq))}Hz"
        base += f"_{duration:g}s_{rate}Hz"
        outfile = base + ".wav"
    outfile = os.path.abspath(outfile)

    # Валидация режимов
    if not args.poly and sig_type is None:
        raise SystemExit("Нужно указать тип сигнала или использовать --poly")
    if args.freqs and sig_type is None:
        raise SystemExit("Для --freqs необходимо указать позиционный тип сигнала")
    # Разрешаем одновременное использование AM и FM: сначала FM, затем AM

    # Полифония одинакового типа: если указан --freqs, генерируем набор голосов и смешиваем
    voices = []
    if args.poly:
        groups = [g.strip() for g in args.poly.split(";") if g.strip()]
        if not groups:
            raise SystemExit("--poly: не указаны группы сигналов")
        for grp in groups:
            parts = [p.strip() for p in grp.split(":", 1)]
            if len(parts) == 1:
                grp_type = parts[0].lower()
                freqs_list = []
            else:
                grp_type = parts[0].lower()
                freqs_list = [x for x in parts[1].split(",") if x]

            if grp_type not in {"sine", "triangle", "saw", "pulse", "noise"}:
                raise SystemExit(f"--poly: неизвестный тип '{grp_type}'")

            if grp_type == "noise":
                # Для шума частоты не нужны — генерируем одну дорожку шума
                voices.append(signals.generate_noise(duration, rate, amp))
                continue

            if not freqs_list:
                raise SystemExit(f"--poly: для типа '{grp_type}' нужно указать частоты")
            try:
                freq_values = [float(x) for x in freqs_list]
            except ValueError:
                raise SystemExit(f"--poly: неверный список частот в группе '{grp}'")

            for f in freq_values:
                if grp_type == "sine":
                    voices.append(signals.generate_sine(f, duration, rate, amp))
                elif grp_type == "triangle":
                    voices.append(signals.generate_triangle(f, duration, rate, amp))
                elif grp_type == "saw":
                    voices.append(signals.generate_sawtooth(f, duration, rate, amp))
                elif grp_type == "pulse":
                    voices.append(signals.generate_pulse(f, duration, duty, rate, amp))
        samples = signals.mix_signals(voices, normalize=(not args.no_normalize))
    elif freqs_arg:
        try:
            freq_values = [float(x.strip()) for x in freqs_arg.split(",") if x.strip()]
        except ValueError:
            raise SystemExit("Неверный формат --freqs. Пример: --freqs 440,550,660")
        for f in freq_values:
            if sig_type == "sine":
                voices.append(signals.generate_sine(f, duration, rate, amp))
            elif sig_type == "triangle":
                voices.append(signals.generate_triangle(f, duration, rate, amp))
            elif sig_type == "saw":
                voices.append(signals.generate_sawtooth(f, duration, rate, amp))
            elif sig_type == "pulse":
                voices.append(signals.generate_pulse(f, duration, duty, rate, amp))
            else:
                raise SystemExit("--freqs недоступен для типа 'noise'")
        samples = signals.mix_signals(voices, normalize=(not args.no_normalize))
    else:
        if sig_type == "sine":
            samples = signals.generate_sine(freq, duration, rate, amp)
        elif sig_type == "triangle":
            samples = signals.generate_triangle(freq, duration, rate, amp)
        elif sig_type == "saw":
            samples = signals.generate_sawtooth(freq, duration, rate, amp)
        elif sig_type == "noise":
            samples = signals.generate_noise(duration, rate, amp)
        elif sig_type == "pulse":
            samples = signals.generate_pulse(freq, duration, duty, rate, amp)
        else:
            raise ValueError("Неизвестный тип сигнала")

    # Применить модуляцию к итоговому сигналу (сначала FM, затем AM)
    if args.fm:
        if sig_type is None:
            raise SystemExit("Для FM требуется указать базовый тип несущей как позиционный аргумент")
        try:
            parts = [p.strip() for p in args.fm.split(":")]
            mod_type = parts[0]
            mod_freq = float(parts[1])
            deviation_hz = float(parts[2])
        except Exception:
            raise SystemExit("--fm формат: mod_type:mod_freq:deviation_hz")
        # Генерируем FM по формуле, т.к. частота меняется во времени
        samples = signals.generate_fm(
            sig_type,
            freq,
            duration,
            rate,
            amp,
            mod_type,
            mod_freq,
            deviation_hz,
            carrier_duty=duty,
            mod_duty=duty,
        )

    if args.am:
        try:
            parts = [p.strip() for p in args.am.split(":")]
            mod_type = parts[0]
            mod_freq = float(parts[1]) if len(parts) > 1 else 1.0
            depth = float(parts[2]) if len(parts) > 2 else 1.0
        except Exception:
            raise SystemExit("--am формат: mod_type:mod_freq:depth")
        # Модулятор в [-1,1]
        mod = signals._generate_modulator(mod_type, mod_freq, duration, rate, duty)
        samples = signals.apply_amplitude_modulation(samples, mod, depth)

    # Применить эффект эха
    if args.echo:
        try:
            parts = [p.strip() for p in args.echo.split(":")]
            delay = float(parts[0])
            decay = float(parts[1]) if len(parts) > 1 else 0.5
            num_echoes = int(parts[2]) if len(parts) > 2 else 3
        except Exception:
            raise SystemExit("--echo формат: delay:decay:echoes. Пример: 0.3:0.5:2")
        samples = signals.apply_echo(samples, delay, decay, rate, num_echoes)

    write_wav_mono(outfile, samples, rate)
    print(f"Сохранено: {outfile}")


if __name__ == "__main__":
    main()


