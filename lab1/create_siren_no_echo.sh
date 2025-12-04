#!/bin/bash

# Скрипт для создания сирен БЕЗ эха
# Использование: ./create_siren_no_echo.sh <тип_сирены> [длительность]

SIREN_TYPE="$1"
DURATION="${2:-3}"

echo "Создание сирены БЕЗ эха типа: $SIREN_TYPE (длительность: ${DURATION}с)"

case "$SIREN_TYPE" in
    "ambulance")
        python3 main.py sine -f 900 -t $DURATION --fm triangle:1.5:300 -o ambulance_siren_no_echo_${DURATION}s.wav
        echo "Создана сирена скорой помощи БЕЗ эха: ambulance_siren_no_echo_${DURATION}s.wav"
        ;;
    "police")
        python3 main.py sine -f 700 -t $DURATION --fm sine:2:250 -o police_siren_no_echo_${DURATION}s.wav
        echo "Создана полицейская сирена БЕЗ эха: police_siren_no_echo_${DURATION}s.wav"
        ;;
    "fire")
        python3 main.py sine -f 600 -t $DURATION --fm triangle:3:400 -o fire_truck_siren_no_echo_${DURATION}s.wav
        echo "Создана сирена пожарной машины БЕЗ эха: fire_truck_siren_no_echo_${DURATION}s.wav"
        ;;
    *)
        echo "Доступные типы сирен БЕЗ эха:"
        echo "  ambulance - сирена скорой помощи"
        echo "  police    - полицейская сирена"
        echo "  fire      - сирена пожарной машины"
        echo ""
        echo "Пример использования:"
        echo "  ./create_siren_no_echo.sh ambulance 4"
        echo "  ./create_siren_no_echo.sh police"
        exit 1
        ;;
esac

echo "Готово!"
