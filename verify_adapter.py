import torch

# Замер памяти
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"Пиковый расход VRAM PyTorch: {peak_vram:.2f} GB")

# Сырой вывод
!nvidia-smi > logs_nvidia_smi.txt
print("Файл logs_nvidia_smi.txt сохранен.")

# ПРОВЕРКА ОБНОВЛЕНИЯ ВЕСОВ (Для раздела Prove Training)
# Извлекаем веса после обучения
trained_lora_weights = {
    k: v.cpu().clone()
    for k, v in model.named_parameters()
    if "lora" in k.lower()
}

# Считаем L2-норму разницы между initial_lora_weights и trained_lora_weights
total_l2_norm = 0.0
for name, init_param in initial_lora_weights.items():
    if name in trained_lora_weights:
        diff = (trained_lora_weights[name].float() - init_param.float())
        total_l2_norm += torch.norm(diff, p=2).item() ** 2

total_l2_norm = total_l2_norm ** 0.5

print(f"\nL2-норма изменения весов LoRA: {total_l2_norm:.6f}")
if total_l2_norm > 1e-5:
    print("[ПРОЙДЕНО] Градиенты применились, веса адаптера изменились.")
else:
    print("[ОШИБКА] Веса заморожены! Модель не обучалась.")