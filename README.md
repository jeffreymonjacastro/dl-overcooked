# 🍳 Tutorial: Overcooked-AI con `uv` y Python 3.10

> **Proyecto:** Overcooked-AI — Recolección de Demostraciones (Entrega 1)  
> **Repositorio oficial:** https://github.com/HumanCompatibleAI/overcooked_ai  
> **Deadline:** 24 de mayo a las 9pm

---

## Índice

1. [Instalación de `uv`](#1-instalación-de-uv)
2. [Clonar el repositorio y crear el entorno](#2-clonar-el-repositorio-y-crear-el-entorno)
3. [Instalación de dependencias](#3-instalación-de-dependencias)
4. [Verificar la instalación](#4-verificar-la-instalación)
5. [Ejecutar el juego (`run_game`)](#5-ejecutar-el-juego-run_game)
6. [Configuración de `collect_demonstrations.yaml`](#6-configuración-de-collect_demonstrationsyaml)
7. [Layouts: encontrar y crear uno personalizado](#7-layouts-encontrar-y-crear-uno-personalizado)
8. [Conclusión: plan de grabaciones](#8-conclusión-plan-de-grabaciones)

---

## 1. Instalación de `uv`

`uv` es un gestor de entornos y paquetes de Python extremadamente rápido, recomendado como alternativa a `venv` + `pip`.

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
```

Verifica la instalación:

```bash
uv --version
```

---

## 2. Descargar el repositorio y crear el entorno

- Crea una carpeta `overcooked`
- Desgarga el zip overcooked.zip en esa carpeta y descomprímelo.
- Se descargarán 3 carpetas: ``scr`, `configs`, `policies`

Crea el entorno virtual con **Python 3.10** (uv lo descarga automáticamente si no lo tienes):

```bash
uv venv --python 3.10
```

Activa el entorno:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

> El prompt debe mostrar `(.venv)` al inicio para confirmar que el entorno está activo.

---

## 3. Instalación de dependencias

Instala en este orden para evitar conflictos de versiones:

```bash
# Dependencias indicadas por el docente
uv pip install overcooked-ai
uv pip install "numpy<2"
uv pip install "PyYAML>=6.0" "Pillow>=10.0" "imageio>=2.31"

# Dependencia adicional requerida por overcooked-ai
uv pip install scipy
```

> ⚠️ **`numpy<2` es obligatorio.** La librería no es compatible con NumPy 2.x y puede fallar si usas una versión más nueva.

> ℹ️ **`scipy`** no aparece en las instrucciones del docente pero es requerida internamente por `overcooked-ai` para sus algoritmos de planificación.

---

## 4. Verificar la instalación

El módulo importable **no** se llama `overcooked_ai` sino `overcooked_ai_py`:

```bash
python -c "import overcooked_ai_py; print('OK')"
```

**Salida esperada:**

```
Gym has been unmaintained since 2022 ...   ← warning inofensivo, ignorar
OK
```

> El warning de `gym` es normal — `overcooked-ai` usa una versión antigua de gym pero no afecta el funcionamiento del proyecto.

También puedes verificar la versión instalada:

```bash
uv pip list | findstr overcooked   # Windows
uv pip list | grep overcooked      # macOS/Linux
```

Debe mostrar `overcooked-ai 1.1.0` (o superior).

---

## 5. Ejecutar el juego (`run_game`)

Este comando abre el juego en modo libre para **probar** que todo funciona. **No guarda grabaciones.**

```bash
python -m src.run_game --config configs/play.yaml
```

Usa el teclado para controlar tu agente. Es útil para practicar antes de grabar.

**Alternativa sin activar el entorno (usando `uv run`):**

```bash
uv run python -m src.run_game --config configs/play.yaml
```

---

## 6. Configuración de `collect_demonstrations.yaml`

Este es el archivo de configuración central para grabar demostraciones. Se encuentra en `configs/collect_demonstrations.yaml`.

```yaml
environment:
  layout_name: cramped_room # ← ESCENARIO a jugar (cambiar por cada grabación)
  horizon: 250 # ← duración en timesteps (NO modificar)

policies:
  agent_0:
    name: greedy_full_task # ← AGENTE AUTOMÁTICO (rotar aleatoriamente)
```

### Variables importantes

| Variable       | Valores posibles                              | Descripción                           |
| -------------- | --------------------------------------------- | ------------------------------------- |
| `layout_name`  | cualquier nombre de layout                    | El escenario que se cargará           |
| `horizon`      | `250` (fijo)                                  | Duración de cada partida en timesteps |
| `agent_0.name` | `stay` / `random_motion` / `greedy_full_task` | Comportamiento del agente automático  |

### Agentes disponibles

| Agente             | Comportamiento                          |
| ------------------ | --------------------------------------- |
| `stay`             | No se mueve, permanece en su lugar      |
| `random_motion`    | Se mueve aleatoriamente sin estrategia  |
| `greedy_full_task` | Intenta completar sopas de forma óptima |

> Asigna el agente **aleatoriamente** en cada grabación, tal como lo indica el docente.

### Carpetas generadas al grabar

```
data/demonstrations/         ← archivos que debes entregar
outputs/collect_demonstrations/  ← logs auxiliares (no entregar)
```

### Comando para grabar

```bash
python -m src.collect_demonstrations --config configs/collect_demonstrations.yaml
```

---

## 7. Layouts: encontrar y crear uno personalizado

### Encontrar los layouts disponibles

Los layouts del paquete instalado están en:

```
.venv\lib\site-packages\overcooked_ai_py\data\layouts\
```

Listarlos:

```bash
# Windows
dir .venv\lib\site-packages\overcooked_ai_py\data\layouts\

# macOS/Linux
ls .venv/lib/python3.10/site-packages/overcooked_ai_py/data/layouts/
```

Ver el contenido de un layout existente (para inspirarte):

```bash
# Windows
type .venv\lib\site-packages\overcooked_ai_py\data\layouts\cramped_room.layout

# macOS/Linux
cat .venv/lib/python3.10/site-packages/overcooked_ai_py/data/layouts/cramped_room.layout
```

**Escenarios disponibles en el repositorio oficial:**

```
asymmetric_advantages   coordination_ring     counter_circuit
cramped_room            forced_coordination   large_room
simple_o                simple_tomato         small_corridor
soup_coordination       tutorial_0            tutorial_1
tutorial_2              tutorial_3
```

### Crear un layout personalizado

Un archivo `.layout` es un mapa de texto plano. Cada carácter representa un elemento del escenario:

| Carácter | Elemento                                   |
| -------- | ------------------------------------------ |
| `X`      | Pared / counter                            |
| ` `      | Suelo (espacio vacío)                      |
| `1`      | Posición inicial del agente 1 (humano)     |
| `2`      | Posición inicial del agente 0 (automático) |
| `O`      | Dispensador de cebollas                    |
| `T`      | Dispensador de tomates                     |
| `D`      | Dispensador de platos                      |
| `P`      | Olla (pot)                                 |
| `S`      | Zona de entrega de sopas                   |

**Ejemplo de layout custom:**

```
XXXXXXXXXXX
X1   P   2X
XO       TX
XD       SX
XXXXXXXXXXX
```

**Pasos para usarlo:**

1. Guarda el archivo como `mi_layout.layout`
2. Cópialo a la carpeta de layouts del paquete:
   ```
   .venv\lib\site-packages\overcooked_ai_py\data\layouts\
   ```
3. En `configs/collect_demonstrations.yaml`, pon:
   ```yaml
   environment:
     layout_name: mi_layout
   ```

> 📌 Si usas layouts custom, recuerda incluir los archivos `.layout` en tu entrega al Drive.

---

## 8. Conclusión: plan de grabaciones

Con todo instalado y configurado, el siguiente paso es completar las **20 grabaciones requeridas**:

- **20 grabaciones en total**
- **10 escenarios distintos**, 2 grabaciones por escenario
- En cada grabación, cambiar `layout_name` y asignar `agent_0.name` aleatoriamente entre `stay`, `random_motion` y `greedy_full_task`
- Cada partida dura **250 timesteps** → en total `250 × 20 = 5000` transiciones humanas

**Ejemplo de plan:**

| #   | `layout_name`           | `agent_0.name`     |
| --- | ----------------------- | ------------------ |
| 1   | `cramped_room`          | `greedy_full_task` |
| 2   | `cramped_room`          | `random_motion`    |
| 3   | `coordination_ring`     | `stay`             |
| 4   | `coordination_ring`     | `greedy_full_task` |
| 5   | `counter_circuit`       | `random_motion`    |
| 6   | `counter_circuit`       | `stay`             |
| 7   | `forced_coordination`   | `greedy_full_task` |
| 8   | `forced_coordination`   | `random_motion`    |
| 9   | `large_room`            | `stay`             |
| 10  | `large_room`            | `greedy_full_task` |
| 11  | `simple_tomato`         | `random_motion`    |
| 12  | `simple_tomato`         | `stay`             |
| 13  | `small_corridor`        | `greedy_full_task` |
| 14  | `small_corridor`        | `random_motion`    |
| 15  | `soup_coordination`     | `stay`             |
| 16  | `soup_coordination`     | `greedy_full_task` |
| 17  | `asymmetric_advantages` | `random_motion`    |
| 18  | `asymmetric_advantages` | `stay`             |
| 19  | `mi_layout` _(custom)_  | `greedy_full_task` |
| 20  | `mi_layout` _(custom)_  | `random_motion`    |

### Checklist de entrega ✅

- [ ] 20 grabaciones en `data/demonstrations/`
- [ ] 10 escenarios distintos (2 grabaciones cada uno)
- [ ] Agente automático asignado aleatoriamente
- [ ] Archivo `integrantes.txt` con los nombres del grupo
- [ ] Archivos `.layout` si usaron escenarios custom
- [ ] Todo subido a la carpeta del grupo en el Drive

---

_Tutorial generado en base al repositorio oficial y sesión de instalación real en Windows con Python 3.10 y uv._
