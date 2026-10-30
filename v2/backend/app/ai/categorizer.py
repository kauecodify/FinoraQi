"""Categorização automática de transações por descrição (TF-IDF + Naive Bayes).

Treinado em um corpus semente em português embutido no código, com possibilidade de
reforço por usuário via `learn`. Retorna categoria e confiança.
"""

from __future__ import annotations

import threading
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

SEED: dict[str, list[str]] = {
    "alimentacao": [
        "ifood pedido",
        "restaurante almoco",
        "padaria pao",
        "supermercado compras",
        "mercado extra",
        "lanchonete",
        "pizzaria jantar",
        "acougue",
        "hortifruti feira",
        "rappi comida",
    ],
    "transporte": [
        "uber viagem",
        "99 corrida",
        "combustivel posto",
        "gasolina shell",
        "estacionamento",
        "metro bilhete",
        "onibus passagem",
        "pedagio sem parar",
        "ipva licenciamento",
        "oficina mecanica",
    ],
    "moradia": [
        "aluguel apartamento",
        "condominio",
        "conta de luz enel",
        "conta de agua sabesp",
        "gas encanado comgas",
        "iptu",
        "internet vivo fibra",
        "manutencao casa",
        "material construcao",
        "seguro residencial",
    ],
    "saude": [
        "farmacia drogasil",
        "consulta medica",
        "plano de saude unimed",
        "exame laboratorio",
        "dentista",
        "academia smartfit",
        "psicologo sessao",
        "hospital",
        "remedio drogaria",
        "otica oculos",
    ],
    "lazer": [
        "netflix assinatura",
        "spotify premium",
        "cinema ingresso",
        "show ingresso",
        "viagem hotel",
        "airbnb hospedagem",
        "steam jogo",
        "bar cerveja",
        "disney plus",
        "passagem aerea latam",
    ],
    "educacao": [
        "mensalidade faculdade",
        "curso online udemy",
        "livraria livro",
        "escola mensalidade",
        "alura assinatura",
        "material escolar",
        "ingles curso",
        "pos graduacao",
        "coursera",
        "apostila concurso",
    ],
    "salario": [
        "salario mensal",
        "pagamento salario empresa",
        "folha de pagamento",
        "adiantamento salarial",
        "decimo terceiro",
        "ferias pagamento",
        "pro labore",
        "bonus anual",
        "plr participacao lucros",
        "freelance projeto",
    ],
    "investimentos": [
        "aplicacao cdb",
        "tesouro direto",
        "compra acoes",
        "fundo imobiliario",
        "rendimento poupanca",
        "dividendos recebidos",
        "corretora xp",
        "resgate cdb",
        "nubank caixinha",
        "previdencia privada",
    ],
    "compras": [
        "amazon compra",
        "mercado livre",
        "shopee pedido",
        "magazine luiza",
        "roupa loja",
        "renner",
        "eletronicos",
        "americanas",
        "presente aniversario",
        "shein pedido",
    ],
    "servicos": [
        "tarifa bancaria",
        "anuidade cartao",
        "celular claro",
        "seguro auto",
        "cartorio",
        "contador honorarios",
        "lavanderia",
        "cabeleireiro",
        "pet shop",
        "assinatura software",
    ],
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


class Categorizer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._texts: list[str] = []
        self._labels: list[str] = []
        for label, examples in SEED.items():
            self._texts += examples
            self._labels += [label] * len(examples)
        self._fit()

    def _fit(self) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
                ("nb", ComplementNB(alpha=0.3)),
            ]
        ).fit(self._texts, self._labels)

    @property
    def categories(self) -> list[str]:
        return sorted(SEED)

    def predict(self, descriptions: list[str]) -> list[dict]:
        if not descriptions:
            return []
        with self._lock:
            proba = self.pipeline.predict_proba(descriptions)
            classes = self.pipeline.classes_
        out = []
        for row in proba:
            idx = int(row.argmax())
            out.append({"category": str(classes[idx]), "confidence": round(float(row[idx]), 4)})
        return out

    def learn(self, description: str, category: str) -> None:
        if category not in SEED:
            raise ValueError(f"categoria desconhecida: {category}")
        with self._lock:
            self._texts.append(description)
            self._labels.append(category)
            self._fit()


_instance: Categorizer | None = None
_instance_lock = threading.Lock()


def get_categorizer() -> Categorizer:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Categorizer()
        return _instance
