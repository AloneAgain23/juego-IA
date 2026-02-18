"""
============================================================
  DINO RUN con Deep Q-Learning (DQN) - PyTorch + Pygame
============================================================
  LIBRERÍAS NECESARIAS (instalar con pip):
  
    pip install pygame torch numpy

  Para GPU (opcional, pero más rápido):
    pip install torch --index-url https://download.pytorch.org/whl/cu118

  CÓMO EJECUTAR:
    python dino_dqn.py

  MODOS:
    - Durante entrenamiento: la ventana muestra el juego en tiempo real
    - Presiona ESC para salir
    - La IA aprende progresivamente (verás mejoras en el Score)
============================================================
"""

import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ─────────────────────────────────────────────
#  HIPERPARÁMETROS (tuning sencillo desde aquí)
# ─────────────────────────────────────────────
LEARNING_RATE   = 0.001      # Qué tan rápido aprende la red neuronal
GAMMA           = 0.99       # Factor de descuento (importancia del futuro)
EPSILON_START   = 1.0        # Exploración inicial (100% aleatoria)
EPSILON_END     = 0.01       # Exploración mínima (casi siempre usa la IA)
EPSILON_DECAY   = 0.995      # Velocidad de reducción de exploración
MEMORY_SIZE     = 10_000     # Tamaño del Replay Buffer
BATCH_SIZE      = 64         # Muestras por paso de entrenamiento
TARGET_UPDATE   = 10         # Cada cuántos episodios se actualiza la red target
TRAIN_SPEED     = 3          # Multiplicador de velocidad del juego (> = más rápido)

# ─────────────────────────────────────────────
#  CONFIGURACIÓN DEL JUEGO
# ─────────────────────────────────────────────
SCREEN_W, SCREEN_H = 800, 300
GROUND_Y           = 250       # Coordenada Y del suelo
FPS                = 60        # Frames por segundo (base)

# Paleta de colores
COLOR_BG       = (40,  40,  40)
COLOR_GROUND   = (100, 100, 100)
COLOR_DINO     = (50,  200, 100)   # Verde para el dino
COLOR_CACTUS   = (220,  80,  80)   # Rojo para cactus
COLOR_TEXT     = (200, 200, 200)
COLOR_SCORE    = (255, 215,   0)   # Dorado para el score


# ══════════════════════════════════════════════════════════════
#  1.  ENTORNO DEL JUEGO  (DinoEnv)
# ══════════════════════════════════════════════════════════════
class DinoEnv:
    """
    Clon simplificado de Chrome Dino Run.

    API estilo Gym:
        state         = env.reset()
        state, reward, done, score = env.step(action)

    Acciones:
        0 → No hacer nada
        1 → Saltar

    Estado (vector de 5 valores normalizados):
        [dist_obstaculo, altura_obstaculo, ancho_obstaculo,
         velocidad_juego, velocidad_y_dino]
    """

    # Dimensiones del dinosaurio
    DINO_W, DINO_H = 40, 50
    DINO_X         = 80          # Posición horizontal fija del dino

    # Física del salto
    JUMP_VEL       = -15         # Velocidad inicial del salto (negativa = arriba)
    GRAVITY        =   1         # Aceleración de gravedad por frame

    # Velocidad del juego
    INIT_SPEED     =  4          # Velocidad inicial de los cactus
    SPEED_INC      =  0.002      # Incremento de velocidad por frame

    # Cactus
    MIN_CACTUS_GAP = 300         # Distancia mínima entre cactus

    def __init__(self, render=True):
        self.render_mode = render
        self.screen      = None
        self.font        = None

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("🦕 Dino DQN - Aprendizaje por Refuerzo")
            self.font       = pygame.font.SysFont("monospace", 18)
            self.font_large = pygame.font.SysFont("monospace", 28, bold=True)
            self.clock      = pygame.time.Clock()

        self.reset()

    def reset(self):
        """Reinicia el entorno y devuelve el estado inicial."""
        # Estado del dinosaurio
        self.dino_y    = GROUND_Y - self.DINO_H   # Posición Y del dino
        self.dino_vel  = 0                          # Velocidad vertical
        self.on_ground = True                       # ¿Está en el suelo?

        # Estado del juego
        self.game_speed = self.INIT_SPEED
        self.score      = 0
        self.steps      = 0

        # Lista de cactus: cada uno es un dict con {x, w, h}
        self.cactus_list = []
        self._spawn_cactus(first=True)

        return self._get_state()

    def _spawn_cactus(self, first=False):
        """Genera un nuevo cactus fuera de la pantalla."""
        w = random.randint(20, 35)
        h = random.randint(40, 70)
        # En el primer cactus, empieza más lejos para darle tiempo al dino
        x = SCREEN_W + random.randint(100, 300) if not first else SCREEN_W + 200
        self.cactus_list.append({"x": x, "w": w, "h": h})

    def _get_state(self):
        """
        Construye el vector de estado que 'verá' la red neuronal.
        Normalizar los valores entre 0 y 1 ayuda al entrenamiento.
        """
        if self.cactus_list:
            # Tomar el cactus más cercano que aún no pasó al dino
            next_c = None
            for c in self.cactus_list:
                if c["x"] + c["w"] > self.DINO_X:
                    next_c = c
                    break
            if next_c is None:
                next_c = self.cactus_list[-1]
        else:
            # Sin cactus en pantalla (no debería ocurrir normalmente)
            next_c = {"x": SCREEN_W, "w": 25, "h": 50}

        dist    = (next_c["x"] - self.DINO_X)   / SCREEN_W      # 0–1
        altura  = next_c["h"]                    / SCREEN_H      # 0–1
        ancho   = next_c["w"]                    / 100           # 0–1
        speed   = (self.game_speed - self.INIT_SPEED) / 20      # 0–1
        vel_y   = self.dino_vel                  / 20           # aprox -1 a 1

        return np.array([dist, altura, ancho, speed, vel_y], dtype=np.float32)

    def step(self, action):
        """
        Ejecuta un paso del juego.

        Args:
            action (int): 0 = quieto, 1 = saltar

        Returns:
            state  (np.array): nuevo vector de estado
            reward (float):    recompensa obtenida
            done   (bool):     True si el dino chocó
            score  (int):      puntuación actual
        """
        self.steps      += 1
        self.score      += 1
        self.game_speed += self.SPEED_INC

        # ── Acción: saltar ──────────────────────────────────────
        if action == 1 and self.on_ground:
            self.dino_vel  = self.JUMP_VEL
            self.on_ground = False

        # ── Física del salto (gravedad) ─────────────────────────
        self.dino_y   += self.dino_vel
        self.dino_vel += self.GRAVITY

        # Suelo: evitar que el dino pase del piso
        ground_limit = GROUND_Y - self.DINO_H
        if self.dino_y >= ground_limit:
            self.dino_y    = ground_limit
            self.dino_vel  = 0
            self.on_ground = True

        # ── Mover cactus ────────────────────────────────────────
        for c in self.cactus_list:
            c["x"] -= self.game_speed

        # Eliminar cactus que salieron por la izquierda
        self.cactus_list = [c for c in self.cactus_list if c["x"] + c["w"] > -10]

        # Generar nuevo cactus si el último ya está suficientemente adentro
        if not self.cactus_list or self.cactus_list[-1]["x"] < SCREEN_W - self.MIN_CACTUS_GAP:
            self._spawn_cactus()

        # ── Detección de colisión ────────────────────────────────
        dino_rect = pygame.Rect(
            self.DINO_X + 5, self.dino_y + 5,    # pequeño margen de tolerancia
            self.DINO_W - 10, self.DINO_H - 5
        )
        done   = False
        reward = 1.0   # Recompensa base por sobrevivir un frame

        for c in self.cactus_list:
            cactus_rect = pygame.Rect(
                c["x"], GROUND_Y - c["h"], c["w"], c["h"]
            )
            if dino_rect.colliderect(cactus_rect):
                reward = -10.0   # Penalización por choque
                done   = True
                break

        state = self._get_state()
        return state, reward, done, self.score

    def render(self, episode, epsilon, best_score):
        """Dibuja un frame del juego en la ventana de Pygame."""
        if not self.render_mode or self.screen is None:
            return

        self.screen.fill(COLOR_BG)

        # ── Suelo ──────────────────────────────────────────────
        pygame.draw.line(self.screen, COLOR_GROUND, (0, GROUND_Y), (SCREEN_W, GROUND_Y), 2)

        # ── Dinosaurio ─────────────────────────────────────────
        pygame.draw.rect(self.screen, COLOR_DINO,
                         (self.DINO_X, self.dino_y, self.DINO_W, self.DINO_H),
                         border_radius=6)
        # Ojo del dino
        pygame.draw.circle(self.screen, COLOR_BG,
                           (self.DINO_X + self.DINO_W - 8, self.dino_y + 10), 4)

        # ── Cactus ─────────────────────────────────────────────
        for c in self.cactus_list:
            pygame.draw.rect(self.screen, COLOR_CACTUS,
                             (c["x"], GROUND_Y - c["h"], c["w"], c["h"]),
                             border_radius=3)

        # ── HUD ────────────────────────────────────────────────
        texts = [
            (f"Episodio : {episode:>5}",    20,  15),
            (f"Score    : {self.score:>5}", 20,  38),
            (f"Mejor    : {best_score:>5}", 20,  61),
            (f"Epsilon  : {epsilon:.3f}",   20,  84),
            (f"Velocidad: {self.game_speed:.1f}", 20, 107),
        ]
        for txt, x, y in texts:
            surf = self.font.render(txt, True, COLOR_TEXT)
            self.screen.blit(surf, (x, y))

        # ── Título ─────────────────────────────────────────────
        title = self.font_large.render("🦕 Dino DQN", True, COLOR_SCORE)
        self.screen.blit(title, (SCREEN_W - 220, 15))

        pygame.display.flip()
        self.clock.tick(FPS * TRAIN_SPEED)

    def close(self):
        if self.render_mode:
            pygame.quit()


# ══════════════════════════════════════════════════════════════
#  2.  RED NEURONAL  (DQNetwork)
# ══════════════════════════════════════════════════════════════
class DQNetwork(nn.Module):
    """
    Red neuronal fully-connected que aproxima la función Q(s, a).

    Arquitectura:
        Input(5) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(2)

    La salida es un vector de Q-values, uno por acción.
    Q(s, 0) = valor estimado de "no hacer nada" en estado s
    Q(s, 1) = valor estimado de "saltar" en estado s
    """

    def __init__(self, input_size=5, hidden_size=128, output_size=2):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,  hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════
#  3.  REPLAY BUFFER  (ExperienceReplay)
# ══════════════════════════════════════════════════════════════
class ReplayBuffer:
    """
    Memoria de experiencias pasadas (Experience Replay).

    Almacena tuplas (s, a, r, s', done) y permite muestrear
    mini-batches aleatorios para romper la correlación temporal.
    Sin esto, la red aprendería secuencias muy correlacionadas
    y el entrenamiento sería inestable.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Guarda una transición en la memoria."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Devuelve un mini-batch aleatorio."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════
#  4.  AGENTE DQN  (DQNAgent)
# ══════════════════════════════════════════════════════════════
class DQNAgent:
    """
    Agente de Deep Q-Learning.

    Combina:
      - Red de política  (policy_net):  la que se entrena en cada paso
      - Red objetivo     (target_net):  copia estable que se actualiza cada N episodios
      - Replay Buffer:                  memoria de experiencias pasadas
      - Política ε-greedy:              balance exploración / explotación
    """

    def __init__(self, state_size=5, action_size=2):
        self.state_size  = state_size
        self.action_size = action_size
        self.epsilon     = EPSILON_START

        # Dispositivo: usa GPU si está disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Usando dispositivo: {self.device}")

        # Dos redes: policy (se entrena) y target (estable, copiada periódicamente)
        self.policy_net = DQNetwork(state_size, 128, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, 128, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()   # La red target nunca se gradúa directamente

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory    = ReplayBuffer(MEMORY_SIZE)

    # ── Selección de acción (ε-greedy) ────────────────────────
    def select_action(self, state):
        """
        Política ε-greedy:
          - Con probabilidad ε: acción aleatoria  (exploración)
          - Con probabilidad 1-ε: la mejor acción según Q  (explotación)
        ε se reduce gradualmente para pasar de explorar a explotar.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)   # Acción aleatoria

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()   # Acción con mayor Q-value

    # ── Guardar experiencia ────────────────────────────────────
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # ── Paso de aprendizaje (ECUACIÓN DE BELLMAN) ──────────────
    def learn(self):
        """
        Actualiza los pesos de la red neuronal usando un mini-batch
        de experiencias pasadas.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ECUACIÓN DE BELLMAN (corazón del DQN):

            Q_target(s, a) = r + γ · max_a' Q_target(s', a')
                             ↑           ↑
                          recompensa   valor futuro descontado
                          inmediata    (según la red TARGET)

        El objetivo es minimizar la diferencia entre:
            Q_policy(s, a)   ← lo que predice nuestra red
            Q_target(s, a)   ← lo que "debería" ser según Bellman

        La pérdida (loss) es el error cuadrático medio (MSE):
            L = MSE( Q_policy(s,a) , Q_target(s,a) )

        Al minimizar L con gradiente descendente, la red aprende
        a asociar cada estado-acción con su valor real esperado.
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        if len(self.memory) < BATCH_SIZE:
            return   # Esperar a tener suficientes experiencias

        # ── 1. Muestrear un mini-batch del replay buffer ───────
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Convertir a tensores de PyTorch
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # ── 2. Calcular Q_policy(s, a) ─────────────────────────
        #   La policy_net predice Q para TODAS las acciones;
        #   usamos .gather() para quedarnos solo con la acción tomada.
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        #   Resultado: tensor de shape [BATCH_SIZE]

        # ── 3. Calcular Q_target(s', a') con la red TARGET ─────
        #   Usamos target_net (sin gradientes) para mayor estabilidad.
        #   Si el episodio terminó (done=1), el valor futuro es 0.
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            # ECUACIÓN DE BELLMAN APLICADA AQUÍ ↓
            q_targets = rewards_t + GAMMA * next_q * (1 - dones_t)

        # ── 4. Calcular la pérdida (Huber Loss = más robusto que MSE) ─
        loss = nn.SmoothL1Loss()(q_values, q_targets)
        #   Huber Loss = MSE si |error| < 1, MAE si no. Evita gradientes explosivos.

        # ── 5. Backpropagation: actualizar pesos de policy_net ─
        self.optimizer.zero_grad()   # Limpiar gradientes del paso anterior
        loss.backward()              # Calcular gradientes (backprop)
        # Clip de gradientes para evitar explosiones
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()        # Actualizar pesos con gradiente descendente

        return loss.item()

    # ── Actualizar la red TARGET ───────────────────────────────
    def update_target(self):
        """
        Copia los pesos de policy_net → target_net.
        Se hace cada TARGET_UPDATE episodios para mantener
        los Q-targets estables durante el entrenamiento.
        Sin esto, estaríamos "persiguiendo un objetivo móvil".
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Reducir epsilon (menos exploración con el tiempo) ─────
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path="dino_dqn.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"  ✅ Modelo guardado en {path}")

    def load(self, path="dino_dqn.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"  ✅ Modelo cargado desde {path}")


# ══════════════════════════════════════════════════════════════
#  5.  BUCLE DE ENTRENAMIENTO PRINCIPAL
# ══════════════════════════════════════════════════════════════
def train():
    print("=" * 60)
    print("  🦕 DINO DQN - Entrenamiento iniciado")
    print("=" * 60)
    print(f"  Episodios máximos : ilimitado (Ctrl+C para parar)")
    print(f"  Batch size        : {BATCH_SIZE}")
    print(f"  Gamma (descuento) : {GAMMA}")
    print(f"  Epsilon inicial   : {EPSILON_START}")
    print(f"  Learning rate     : {LEARNING_RATE}")
    print(f"  Velocidad render  : x{TRAIN_SPEED}")
    print("=" * 60 + "\n")

    env       = DinoEnv(render=True)
    agent     = DQNAgent(state_size=5, action_size=2)

    episode        = 0
    best_score     = 0
    scores_history = deque(maxlen=100)   # Últimos 100 scores para calcular promedio
    running        = True

    while running:
        # ── Inicio de episodio ─────────────────────────────────
        state = env.reset()
        done  = False
        total_loss = 0
        steps_loss = 0

        while not done:
            # ── Manejar eventos de Pygame (cerrar ventana, ESC) ─
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done    = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    done    = True

            if not running:
                break

            # ── El agente elige una acción ─────────────────────
            action = agent.select_action(state)

            # ── El entorno ejecuta la acción ────────────────────
            next_state, reward, done, score = env.step(action)

            # ── Guardar la experiencia en el replay buffer ──────
            agent.remember(state, action, reward, next_state, done)

            # ── Aprender de las experiencias pasadas ────────────
            loss = agent.learn()
            if loss is not None:
                total_loss += loss
                steps_loss += 1

            state = next_state

            # ── Renderizar el frame ─────────────────────────────
            env.render(episode, agent.epsilon, best_score)

        # ── Fin del episodio ───────────────────────────────────
        if not running:
            break

        episode += 1
        scores_history.append(env.score)
        avg_score  = np.mean(scores_history)
        best_score = max(best_score, env.score)
        avg_loss   = total_loss / steps_loss if steps_loss > 0 else 0

        # Reducir epsilon gradualmente
        agent.decay_epsilon()

        # Actualizar la red target cada TARGET_UPDATE episodios
        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        # Guardar el mejor modelo
        if env.score >= best_score:
            agent.save("dino_dqn_best.pth")

        # ── Imprimir estadísticas ──────────────────────────────
        print(
            f"  Ep {episode:>4} | "
            f"Score: {env.score:>5} | "
            f"Mejor: {best_score:>5} | "
            f"Avg100: {avg_score:>6.1f} | "
            f"ε: {agent.epsilon:.3f} | "
            f"Loss: {avg_loss:.4f}"
        )

    # ── Fin del entrenamiento ──────────────────────────────────
    print("\n  Entrenamiento finalizado.")
    agent.save("dino_dqn_final.pth")
    env.close()


# ══════════════════════════════════════════════════════════════
#  6.  MODO DEMO: ver la IA ya entrenada jugar
# ══════════════════════════════════════════════════════════════
def demo(model_path="dino_dqn_best.pth"):
    """
    Carga un modelo guardado y lo muestra jugar sin entrenar.
    Ejecutar: demo()  (o modificar el __main__ de abajo)
    """
    print(f"  🎮 Modo DEMO - cargando {model_path}")
    env   = DinoEnv(render=True)
    agent = DQNAgent(state_size=5, action_size=2)
    agent.load(model_path)
    agent.epsilon = 0.0   # Sin exploración en demo

    for ep in range(10):
        state = env.reset()
        done  = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            action          = agent.select_action(state)
            state, _, done, score = env.step(action)
            env.render(ep + 1, 0.0, score)

        print(f"  Demo episodio {ep+1} | Score: {score}")

    env.close()


# ══════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Para entrenar la IA:
    train()

    # Para ver la IA jugar (después de entrenar), comenta train() y usa:
    # demo("dino_dqn_best.pth")
