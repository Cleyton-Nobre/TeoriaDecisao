import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Leitura dos dados
equipDB = pd.read_csv("EquipDB.csv", header=None, names=["ID_equipamento", "TempoAposFalha", "Cluster", "CustoDeFalha"])
clusterDB = pd.read_csv("ClusterDB.csv", header=None, names=["ID_Cluster", "eta", "beta"])
MPDB = pd.read_csv("MPDB.csv", header=None, names=["ID_plano_risco", "Fator_risco(k)", "CustoDoPlano"])

equipDB = equipDB.merge(clusterDB, left_on="Cluster", right_on="ID_Cluster")

# Função de falha de Weibull
def Fi(t, eta, beta):
    return 1 - np.exp(-(t / eta) ** beta)

# Probabilidade condicional de falha
def P(t0, eta, beta, k, delta_t):
    return (Fi((t0 + k * delta_t), eta=eta, beta=beta) - Fi(t=t0, eta=eta, beta=beta)) / (1 - Fi(t=t0, eta=eta, beta=beta))

# Criar a matriz  de probabilidade Pij
Pij = np.zeros((len(equipDB), len(MPDB)))
delta_t = 5  # Definir um valor para Delta_t

for i, equipamento in equipDB.iterrows():
    t0 = equipamento["TempoAposFalha"]
    eta = equipamento["eta"]
    beta = equipamento["beta"]
    for j, plano in MPDB.iterrows():
        k = plano["Fator_risco(k)"]
        Pij[i, j] = P(t0, eta, beta, k, delta_t)

# Converter a matriz para um DataFrame para melhor visualização
Pij_df = pd.DataFrame(Pij, index=equipDB["ID_equipamento"], columns=MPDB["ID_plano_risco"])

# Gerar Xij inicial com base na criticidade
def gerar_Xij_inicial(equipDB, MPDB, Pij):
    """Gera uma solução inicial balanceada para Xij."""
    n_equip = len(equipDB)
    Xij = np.zeros((n_equip, len(MPDB)))

    # Critério de criticidade: custo_falha * probabilidade sem manutenção
    criticidade = equipDB["CustoDeFalha"].values * Pij[:, 0] * equipDB["TempoAposFalha"]
    equip_ordenados = np.argsort(-criticidade)  # Ordem decrescente

    # Distribuição 20% detalhada, 30% intermediária, 50% nenhuma
    for idx, i in enumerate(equip_ordenados):
        if idx < 0.2 * n_equip:
            Xij[i, 2] = 1  # Plano 3 (detalhada)
        elif idx < 0.5 * n_equip:
            Xij[i, 1] = 1  # Plano 2 (intermediária)
        else:
            Xij[i, 0] = 1  # Plano 1 (nenhuma)

    return Xij

Xij = gerar_Xij_inicial(equipDB=equipDB, MPDB=MPDB, Pij=Pij)

# Operadores de vizinhança
def move(X):
    i = random.randint(0, len(X) - 1)
    plano_atual = int(X[i].argmax())
    novo_plano = random.choice([p for p in range(len(X[0])) if p != plano_atual])
    X[i][plano_atual] = 0
    X[i][novo_plano] = 1

def one_opt(X):
    for i in range(10):
        par = random.sample(range(len(X)), 2)
        if not np.array_equal(X[par[0]], X[par[1]]):
            X[par[0]], X[par[1]] = X[par[1]].copy(), X[par[0]].copy()
            break

def cycle_shift(X):
    for _ in range(10):
        i1, i2, i3 = random.sample(range(len(X)), 3)
        if len({tuple(X[i1]), tuple(X[i2]), tuple(X[i3])}) > 1:
            break

    p1 = int(X[i1].argmax())
    p2 = int(X[i2].argmax())
    p3 = int(X[i3].argmax())

    print(X[i1], X[i2], X[i3])

    X[i1][p1], X[i1][p2] = 0, 1
    X[i2][p2], X[i2][p3] = 0, 1
    X[i3][p3], X[i3][p1] = 0, 1

# Funções Objetivo para vetor de solução

def f1(solucao, MPDB):
    return sum(MPDB.iloc[p - 1]["CustoDoPlano"] for p in solucao)

def f2(solucao, equipDB, MPDB, Pij):
    return sum(
        Pij[i, solucao[i] - 1] * equipDB.loc[i, "CustoDeFalha"]
        for i in range(len(solucao))
    )

# Vizinhanças para vetor de solução
def vizinhanca_1(solucao):
    nova = solucao.copy()
    i = np.random.randint(len(solucao))
    nova[i] = np.random.choice([x for x in [1, 2, 3] if x != solucao[i]])
    return nova

def vizinhanca_2(solucao):
    nova = solucao.copy()
    i, j = np.random.choice(len(solucao), 2, replace=False)
    for idx in [i, j]:
        nova[idx] = np.random.choice([x for x in [1, 2, 3] if x != solucao[idx]])
    return nova

def vizinhanca_3(solucao):
    nova = solucao.copy()
    if len(solucao) < 10:
        return vizinhanca_1(solucao)
    i = np.random.randint(0, len(solucao) - 10)
    for j in range(i, i + 10):
        nova[j] = np.random.choice([x for x in [1, 2, 3] if x != solucao[j]])
    return nova

vizinhas = [vizinhanca_1, vizinhanca_2, vizinhanca_3]

# Heurística construtiva baseada em criticidade
def gerar_solucao_inicial(equipDB, Pij):
    criticidade = equipDB["CustoDeFalha"].values * Pij[:, 0] * equipDB["TempoAposFalha"].values
    ordem = np.argsort(-criticidade)
    solucao = np.ones(len(equipDB), dtype=int)
    n = len(equipDB)
    for idx, i in enumerate(ordem):
        if idx < 0.2 * n:
            solucao[i] = 3
        elif idx < 0.5 * n:
            solucao[i] = 2
        else:
            solucao[i] = 1
    return solucao

# Busca local

def busca_local(solucao, obj_func, *args):
    melhor = solucao.copy()
    melhor_valor = obj_func(melhor, *args)
    for viz in vizinhas:
        candidato = viz(melhor)
        valor = obj_func(candidato, *args)
        if valor < melhor_valor:
            return candidato
    return melhor

# GVNS

def GVNS(obj_func, equipDB, MPDB, Pij=None, max_iter=100, return_curve=False):
    if obj_func == f2:
        assert Pij is not None, "Pij é obrigatória para a função f2."
    s = gerar_solucao_inicial(equipDB, Pij)
    melhor = s.copy()
    historico = []
    valor_inicial = obj_func(melhor, MPDB) if obj_func == f1 else obj_func(melhor, equipDB, MPDB, Pij)
    historico.append(valor_inicial)
    for _ in range(max_iter):
        k = 0
        while k < len(vizinhas):
            s_ = vizinhas[k](melhor)
            s__ = busca_local(s_, obj_func, MPDB) if obj_func == f1 else busca_local(s_, obj_func, equipDB, MPDB, Pij)
            novo_valor = obj_func(s__, MPDB) if obj_func == f1 else obj_func(s__, equipDB, MPDB, Pij)
            if novo_valor < historico[-1]:
                melhor = s__.copy()
                historico.append(novo_valor)
                k = 0
            else:
                k += 1
                historico.append(historico[-1])
    return (melhor, historico) if return_curve else melhor

# Execução principal com 5 repetições
if __name__ == "__main__":
    resultados_f1 = [GVNS(f1, equipDB, MPDB, Pij, return_curve=True) for _ in range(5)]
    resultados_f2 = [GVNS(f2, equipDB, MPDB, Pij, return_curve=True) for _ in range(5)]

    custos_f1 = [f1(sol, MPDB) for sol, _ in resultados_f1]
    custos_f2 = [f2(sol, equipDB, MPDB, Pij) for sol, _ in resultados_f2]

    print("--- f1 (Custo de Manutenção) ---")
    print("Min:", np.min(custos_f1))
    print("Max:", np.max(custos_f1))
    print("Std:", np.std(custos_f1))

    print("\n--- f2 (Custo Esperado de Falha) ---")
    print("Min:", np.min(custos_f2))
    print("Max:", np.max(custos_f2))
    print("Std:", np.std(custos_f2))

    plt.figure()
    for _, hist in resultados_f1:
        plt.plot(hist, alpha=0.7)
    plt.title("Convergência GVNS - f1")
    plt.xlabel("Iterações")
    plt.ylabel("Custo de Manutenção")
    plt.grid(True)
    plt.show()

    plt.figure()
    for _, hist in resultados_f2:
        plt.plot(hist, alpha=0.7)
    plt.title("Convergência GVNS - f2")
    plt.xlabel("Iterações")
    plt.ylabel("Custo Esperado de Falha")
    plt.grid(True)
    plt.show()