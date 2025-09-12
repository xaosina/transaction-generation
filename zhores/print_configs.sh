#!/bin/bash
#
# list_configs.sh — печатает все конфиги в директории

CONFIG_DIR=${1:-zhores/configs}

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Ошибка: директория '$CONFIG_DIR' не найдена"
  exit 1
fi

echo "Найденные конфиги:"
for cfg in "$CONFIG_DIR"/age/*.yaml; do
  echo "  - $(basename "$cfg")"
done
