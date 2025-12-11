Horus 👁️
Sistema de Detecção e Classificação Multi-Label de Discurso de Ódio em Português.

O Horus é um framework robusto desenvolvido para o fine-tuning de modelos de linguagem (LLMs) voltados para a moderação de conteúdo. Ele utiliza o estado da arte em Processamento de Linguagem Natural (PLN) para identificar múltiplas categorias de toxicidade em um único texto, utilizando arquiteturas baseadas em Transformer (BERT).

📋 Funcionalidades
Classificação Multi-Label: Capaz de detectar múltiplas tags simultaneamente (ex: Racismo, Misoginia, Homofobia, Gordofobia, etc.).

Fine-Tuning Estável: Pipeline de treinamento otimizado utilizando Weighted Binary Cross-Entropy (BCE) para lidar com datasets desbalanceados sem instabilidade numérica.

Thresholding Adaptativo: Cálculo automático do limiar (threshold) ideal para cada classe individualmente, maximizando o F1-Score.

Interface Gráfica (GUI): Integração com PySide6 (Qt) para gerenciamento de treinos e inferência.

Explicabilidade (Backend): Estrutura preparada para integração com SHAP e LIME, permitindo a análise da importância dos tokens na classificação.

🛠️ Tecnologias Utilizadas
Core: Python 3

Deep Learning: PyTorch, Transformers (Hugging Face)

Modelo Base: neuralmind/bert-base-portuguese-cased (BERTimbau)

Interface: PySide6

Métricas & Dados: Scikit-learn, Pandas, NumPy

Explicabilidade: SHAP, LIME
