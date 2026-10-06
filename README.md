# Hoja de trabajo #2 — Transformers y mecanismos de atención

**CC3092 – Deep Learning y Sistemas Inteligentes** · Universidad del Valle de Guatemala
**Autor:** Renato Rojas (23813)

Análisis de los mecanismos de atención: verificaciones numéricas de la *scaled dot-product attention*, uso de las APIs de atención de PyTorch, medición del costo computacional y visualización de la atención en **BERT** (`bert-base-uncased`) y **GPT-2** (`gpt2`).

## Contenido del repositorio

| Archivo / carpeta | Descripción |
|---|---|
| `Hoja2_Transformers.ipynb` | Notebook completo y comentado con todos los experimentos (partes A–H). |
| `Hoja2_Parte_Teorica.pdf` | Informe (máx. 5 páginas): investigación, resultados, discusión, conclusiones y referencias. |
| `figures/` | Figuras generadas por el notebook (`R1`–`R4` son las versiones compactas usadas en el informe). |
| `results/` | Tablas en CSV: costo de la atención, perfiles de cabezas, análisis de «it» y entropía por capa. |
| `requirements.txt` | Dependencias de Python. |

## Estructura del notebook

| Parte | Contenido |
|---|---|
| A | Verificaciones numéricas: `Var(q·k) = d_k`, saturación de la softmax sin `1/√d_k`, parámetros de multi-head, equivarianza a permutaciones y codificación sinusoidal. |
| B | `F.scaled_dot_product_attention`, `nn.MultiheadAttention` (máscaras y pesos) y `nn.TransformerEncoderLayer` (Post-LN vs Pre-LN). |
| C | Tiempo y memoria de la self-attention en función de la longitud `n`. |
| D | Mapas de calor de la atención en BERT y GPT-2 y perfil de las 144 cabezas de cada modelo. |
| E | Desambiguación de «it» (*animal* vs *street*) en BERT, con prueba de robustez en 6 pares de oraciones. |
| F | GPT-2: verificación de la máscara causal y *attention sinks*. |
| G | Entropía de la atención por capa y por cabeza (BERT vs GPT-2). |
| H | (Opcional) visualización interactiva con `bertviz`. |

## Principales resultados

- **Escalamiento:** `Var(q·k) ≈ d_k`. Sin el factor `1/√d_k` la softmax se satura (`p_max = 0.96` con `d_k = 1024`) y la norma de su jacobiano es 6× menor.
- **Costo:** pendiente log-log de **2.10** en tiempo y **1.90** en memoria para `n ≥ 1024` (el valor teórico es 2). `F.scaled_dot_product_attention` es ≈ 2× más rápida, pero también cuadrática.
- **Especialización de cabezas:** hay cabezas posicionales casi perfectas (token siguiente/anterior), cabezas que atienden a `[SEP]` y a la puntuación, y una cabeza de correferencia (BERT, capa 7·cabeza 11) que resuelve «it» en 10 de 12 oraciones.
- **Attention sinks:** en GPT-2 el primer token recibe hasta el 74 % de la atención (capa 8), sin importar qué palabra sea.
- **Entropía:** en ambos modelos la atención es más concentrada en las capas profundas que en las iniciales.

## Cómo ejecutarlo

Requiere Python 3.11. `requirements.txt` instala PyTorch con CUDA 12.6. El notebook también funciona en CPU, solo que el benchmark de la parte C usa secuencias más cortas.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
jupyter notebook Hoja2_Transformers.ipynb
```

La primera ejecución necesita conexión a internet para descargar BERT y GPT-2 desde Hugging Face (~1 GB en total). Los modelos se cargan con `attn_implementation="eager"`, ya que las implementaciones `sdpa`/FlashAttention no devuelven los pesos de atención.

Los resultados del informe se obtuvieron con PyTorch 2.14.1 + CUDA 12.6 y Transformers 5.18.0 en una NVIDIA GeForce RTX 4060 Laptop GPU. Los tiempos de la parte C varían un poco entre ejecuciones.

## Referencia principal

Vaswani, A. et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). El resto de las referencias está en el informe.
