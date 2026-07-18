import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual-блок (Skip Connection).
    
    Суть: вход x проходит через два слоя (Linear → BatchNorm → ReLU → Linear → BatchNorm),
    а затем к результату ПРИБАВЛЯЕТСЯ исходный x. Это решает проблему затухания градиентов
    и позволяет строить очень глубокие сети (20+ слоёв), не теряя способности учиться.
    
    Именно эта архитектура лежит в основе AlphaZero (DeepMind).
    """
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        residual = x                          # Запоминаем вход (skip connection)
        out = F.relu(self.bn1(self.fc1(x)))   # Первый слой + нормализация + активация
        out = self.bn2(self.fc2(out))          # Второй слой + нормализация
        out = out + residual                   # КЛЮЧ: прибавляем исходный вход
        out = F.relu(out)                      # Финальная активация
        return out


class DurakNet(nn.Module):
    """
    AlphaZero-style нейросеть для игры в Дурак.
    
    Архитектура:
      1. Входной слой: проецирует 220 признаков в скрытое пространство (512 нейронов)
      2. Башня из 10 Residual-блоков: глубокий анализ позиции (20 слоёв суммарно)
      3. Policy Head: предсказывает вероятности для каждого из 74 возможных ходов
      4. Value Head: предсказывает шансы на победу (от -1.0 до +1.0)
    
    На RTX 4060 Ti (8 ГБ) эта модель займёт ~300-400 МБ видеопамяти.
    """
    def __init__(self, input_size=220, hidden_size=512, num_res_blocks=10, action_size=74):
        super(DurakNet, self).__init__()
        
        # === Входной слой: проекция из 220 признаков в скрытое пространство ===
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        
        # === Башня Residual-блоков (сердце AlphaZero) ===
        # 10 блоков × 2 слоя = 20 скрытых слоёв глубины
        self.res_tower = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(num_res_blocks)]
        )
        
        # === Policy Head (Голова политики) ===
        # Отвечает на вопрос: "Какой ход лучше всего сделать?"
        self.policy_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.policy_bn  = nn.BatchNorm1d(hidden_size // 2)
        self.policy_fc2 = nn.Linear(hidden_size // 2, action_size)
        
        # === Value Head (Голова оценки) ===
        # Отвечает на вопрос: "Кто сейчас выигрывает?"
        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.value_bn  = nn.BatchNorm1d(hidden_size // 4)
        self.value_fc2 = nn.Linear(hidden_size // 4, 1)

    def forward(self, x):
        # x: [batch_size, 220] — закодированное состояние игры
        
        # 1. Входная проекция
        x = F.relu(self.input_bn(self.input_fc(x)))
        
        # 2. Прогоняем через башню Residual-блоков
        #    Здесь сеть "думает" — извлекает глубокие закономерности
        x = self.res_tower(x)
        
        # 3. Policy Head — вероятности ходов (сырые логиты)
        p = F.relu(self.policy_bn(self.policy_fc1(x)))
        p = self.policy_fc2(p)
        
        # 4. Value Head — оценка позиции от -1 (проигрыш) до +1 (победа)
        v = F.relu(self.value_bn(self.value_fc1(x)))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

