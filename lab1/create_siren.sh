#!/bin/bash

# Скрипт для быстрого создания сирен экстренных служб
# Использование: ./create_siren.sh <тип_сирены> [длительность]

SIREN_TYPE="$1"
DURATION="${2:-3}"

echo "Создание сирены типа: $SIREN_TYPE (длительность: ${DURATION}с)"

case "$SIREN_TYPE" in
    "ambulance")
        python3 main.py sine -f 900 -t $DURATION --fm triangle:1.5:300 --echo 0.05:0.4:3 -o ambulance_siren_${DURATION}s.wav
        echo "Создана сирена скорой помощи: ambulance_siren_${DURATION}s.wav"
        ;;
    "police")
        python3 main.py sine -f 700 -t $DURATION --fm sine:2:250 --echo 0.08:0.3:2 -o police_siren_${DURATION}s.wav
        echo "Создана полицейская сирена: police_siren_${DURATION}s.wav"
        ;;
    "fire")
        python3 main.py sine -f 600 -t $DURATION --fm triangle:3:400 --echo 0.03:0.5:4 -o fire_truck_siren_${DURATION}s.wav
        echo "Создана сирена пожарной машины: fire_truck_siren_${DURATION}s.wav"
        ;;
    "hilo")
        python3 main.py sine --poly "sine:800,1000" -t $DURATION --fm triangle:1:100 --echo 0.1:0.3:2 -o hi_lo_ambulance_${DURATION}s.wav
        echo "Создана Hi-Lo сирена: hi_lo_ambulance_${DURATION}s.wav"
        ;;
    *)
        echo "Доступные типы сирен:"
        echo "  ambulance - сирена скорой помощи"
        echo "  police    - полицейская сирена"
        echo "  fire      - сирена пожарной машины"
        echo "  hilo      - двухтональная Hi-Lo сирена"
        echo ""
        echo "Пример использования:"
        echo "  ./create_siren.sh ambulance 4"
        echo "  ./create_siren.sh police"
        exit 1
        ;;
esac

echo "Готово!"
