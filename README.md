# Hugging Face - Curso Completo 🚀

Este repositório contém os materiais e exemplos práticos do curso sobre **Hugging Face**, abordando desde a exploração de modelos pré-treinados até a escalabilidade no treinamento de modelos.

## 📌 Conteúdo do Curso

### 1️⃣ Explorando Modelos Pré-Treinados
Neste passo inicial, você será introduzido ao Hugging Face, uma das principais plataformas de inteligência artificial, focada em modelos de processamento de linguagem natural. 

- Introdução ao Hugging Face
- Utilização de modelos pré-treinados para NLP e visão computacional
- Hands-on com pipelines e inferência

### 2️⃣ Otimizando Modelos Pré-Treinados
Aqui você aprenderá técnicas avançadas para melhorar modelos de linguagem de larga escala.

- Transfer learning e fine-tuning
- Uso do Optimum para otimização de modelos
- Customização de modelos para tarefas específicas

### 3️⃣ Escalando o Treinamento de Modelos
No passo final, exploramos técnicas para treinar modelos de forma eficiente em grandes infraestruturas.

- Introdução ao Hugging Face Accelerate
- Treinamento distribuído em diferentes configurações de hardware
- Estratégias para escalar modelos de NLP

---

## 📦 Instalação

Antes de começar, certifique-se de ter o Python instalado. Você pode instalar as bibliotecas necessárias com:

```bash
pip install transformers torch datasets accelerate optimum
```

Se precisar de aceleração por GPU, instale o **CUDA** para PyTorch seguindo as instruções no site oficial do PyTorch.

---

## 🚀 Exemplos de Código

### 🔹 Explorando Modelos Pré-Treinados

```python
from transformers import pipeline

# Criando um pipeline de análise de sentimento
sentiment_pipeline = pipeline("sentiment-analysis")

# Testando o modelo
result = sentiment_pipeline("Hugging Face é incrível!")
print(result)
```

### 🔹 Otimizando Modelos Pré-Treinados

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenizando um texto
inputs = tokenizer("Hugging Face facilita o NLP!", return_tensors="pt")
outputs = model(**inputs)

# Obtendo previsões
logits = outputs.logits
predicted_class = torch.argmax(logits).item()
print(f"Classe prevista: {predicted_class}")
```

### 🔹 Escalando o Treinamento de Modelos

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from accelerate import Accelerator

# Configuração de aceleração
accelerator = Accelerator()

# Carregar dataset
raw_datasets = load_dataset("imdb")

# Definir argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["test"]
)

# Treinar o modelo com aceleração
with accelerator:
    trainer.train()
```

---

## 📖 Documentação Oficial

Para mais detalhes sobre a biblioteca **Transformers**, consulte a documentação oficial: [Hugging Face Docs](https://huggingface.co/docs/transformers/index)

## 📜 Licença

Este projeto segue a licença MIT. Sinta-se à vontade para contribuir! 🚀
