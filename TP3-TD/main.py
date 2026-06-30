import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
import time


equipDB = pd.read_csv("EquipDB.csv", header=None, names=["ID_equipamento", "TempoAposFalha", "Cluster", "CustoDeFalha"])
ClusterDB= pd.read_csv("ClusterDB.csv", header=None, names=["ID_Cluster", "eta", "beta"])
MPDB=pd.read_csv("MPDB.csv", header=None, names=["ID_plano_risco", "Fator_risco(k)", "CustoDoPlano"])

# Unir informações do ClusterDB ao equipDB
equipDB = equipDB.merge(ClusterDB, left_on="Cluster", right_on="ID_Cluster")



def Fi(t, eta, beta):
    return 1-np.exp(-(t/eta)**beta)

def P(t0, eta, beta, k, delta_t):
    return (Fi((t0+k*delta_t), eta=eta, beta=beta)-Fi(t=t0, eta=eta, beta=beta))/(1-Fi(t=t0, eta=eta, beta=beta))


# Criar a matriz  de probabilidade Pij
Pij = np.zeros((len(equipDB), len(MPDB)))
delta_t = 5  # Definir um valor para Delta_t,


for i, equipamento in equipDB.iterrows():
    t0 = equipamento["TempoAposFalha"]
    eta = equipamento["eta"]
    beta = equipamento["beta"]
    
    for j, plano in MPDB.iterrows():
        k = plano["Fator_risco(k)"]
        Pij[i, j] = P(t0, eta, beta, k, delta_t)



# Heurística construtiva baseada em criticidade
def gerar_solucao_inicial(equipDB, MPDB, Pij):
    criticidade = equipDB["CustoDeFalha"].values * Pij[:, 0] 
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

def f1(X, equipDB, MPDB, Pij):
    """
    Calcula o custo total de manutenção.

    Parâmetros:
    - x: matriz numpy (n x 3) representando as variáveis de decisão x_{i,j}
    - m: vetor numpy (3,) representando os custos de manutenção m_j para cada coluna

    Retorna:
    - Custo total de manutenção (float)
    """
    return sum(MPDB.iloc[p - 1]["CustoDoPlano"] for p in X)

def f2(X, equipDB, MPDB, Pij):
    """
    Calcula o custo esperado de falha.

    Parâmetros:
    - x: matriz numpy (n x 3) representando as variáveis de decisão x_{i,j}
    - p: matriz numpy (n x 3) representando as probabilidades de falha p_{i,j}
    - f: vetor numpy (n,) representando os custos de falha f_i

    Retorna:
    - Custo esperado de falha (float)
    """
    return sum(
        Pij[i, X[i] - 1] * equipDB.loc[i, "CustoDeFalha"]
        for i in range(len(X))
    )

one = np.ones(500, dtype=int)
min_f1 = f1(one,equipDB,MPDB,Pij)
max_f1 = f1(one+2,equipDB,MPDB,Pij)
min_f2 = f2(one+2,equipDB,MPDB,Pij)
max_f2 = f2(one,equipDB,MPDB,Pij)


def move(X):
    nova = X.copy()
    seed = int(time.time_ns() % (2**32))
    np.random.seed(seed)
    i = np.random.randint(len(X))
    nova[i] = np.random.choice([x for x in [1, 2, 3] if x != X[i]])
    return nova

# Vizinhança 2 - 1-opt entre planos
def double_move(X):
    nova = X.copy()
    seed = int(time.time_ns() % (2**32))
    np.random.seed(seed)
    i, j = np.random.choice(len(X), 2, replace=False)
    for idx in [i, j]:
        nova[idx] = np.random.choice([x for x in [1, 2, 3] if x != X[idx]])
    return nova

def block_change(X):
    nova = X.copy()
    seed = int(time.time_ns() % (2**32))
    np.random.seed(seed)
    if len(X) < 50:
        return move(X)
    i = np.random.randint(0, len(X) - 50)
    for j in range(i, i + 50):
        nova[j] = np.random.choice([x for x in [1, 2, 3] if x != X[j]])
    return nova

vizinhas = [move, double_move, block_change]

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


def GVNS(obj_func, gerar_solucao_inicial, vizinhas, busca_local, *args, max_iter=1000, return_curve=False):
    """
    Executa o algoritmo GVNS (Variable Neighborhood Search).
    """
    s = gerar_solucao_inicial(*args)
    melhor = s.copy()
    historico = []

    valor_inicial = obj_func(melhor, *args)
    historico.append(valor_inicial)

    for _ in range(max_iter):
        k = 0
        while k < len(vizinhas):
            s_ = vizinhas[k](melhor)
            s__ = busca_local(s_, obj_func, *args)
            novo_valor = obj_func(s__, *args)
            if novo_valor < historico[-1]:
                melhor = s__.copy()
                historico.append(novo_valor)
                k = 0
            else:
                k += 1
                historico.append(historico[-1])

    return (melhor, historico) if return_curve else melhor


# --- Normalização da função objetivo ---
def normfObj(fobj, x, min_, max_, *args):
    """
    Normaliza o valor da função objetivo.
    """
    return (fobj(x, *args) - min_) / (max_ - min_)


# --- Função fábrica para criar a função objetivo com um W específico ---
def make_objf_Pw(W_local):
    """
    Retorna uma função objetivo ponderada com um valor W específico.
    """
    def objf_Pw(x, *args):
        return W_local * normfObj(f1, x, min_f1, max_f1, *args) + (1 - W_local) * normfObj(f2, x, min_f2, max_f2, *args)
    return objf_Pw


# --- Wrapper para execução paralela ---
def run_gvns_wrapper_Pw(_):
    """
    Wrapper para a execução paralela do GVNS, retornando a solução e o peso W.
    """
    try:
        
        seed = int(time.time_ns() % (2**32))
        np.random.seed(seed)
        vizinhancas = [move, double_move, block_change]
        args = (equipDB, MPDB, Pij)
        W_local = random.uniform(0, 1)
        obj_func = make_objf_Pw(W_local)
        resultado = GVNS(obj_func, gerar_solucao_inicial, vizinhancas, busca_local, *args, max_iter=1000, return_curve=True)
        return resultado, W_local  # Retorna a solução e o W utilizado

    except Exception as e:
        print(f"[ERRO] Subprocesso falhou: {e}")
        return None


def executar_uma_repeticao_Pw():
    with ProcessPoolExecutor(max_workers=10) as executor:
        resultados_com_pesos = list(executor.map(run_gvns_wrapper_Pw, range(it)))
    return resultados_com_pesos

def filtrar_pareto_nao_dominado(pontos):
    """Retorna apenas os pontos não dominados (fronteira de Pareto)"""
    pontos = np.array(pontos)
    n = pontos.shape[0]
    is_dominado = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Dominância: queremos minimizar f1 e f2
                if (pontos[j][0] <= pontos[i][0] and pontos[j][1] <= pontos[i][1]) and \
                   (pontos[j][0] < pontos[i][0] or pontos[j][1] < pontos[i][1]):
                    is_dominado[i] = True
                    break
    return pontos[~is_dominado]


def epsilons_f1():

    return np.random.random()*(max_f1-min_f1) + min_f1

def epsilons_f2():
    
    return np.random.random()*(max_f2-min_f2) + min_f2

epsilons = [epsilons_f2,epsilons_f1]

new_f_obj = [[f1,f2],[f2,f1]]

def make_objf_Pe(epsilon,num_func_obj=0, penalidade=100):
    f_obj = new_f_obj[num_func_obj][0]
    f_desig = new_f_obj[num_func_obj][1]

    def objf_Pe(x, *args):
        return f_obj(x, *args) + penalidade*max(f_desig(x, *args) - epsilon,0)
    return objf_Pe


def epsilon_restrito(_): # num_func_obj tem que ser 0 ou 1
    seed = int(time.time_ns() % (2**32))
    np.random.seed(seed)
    
    num_func_obj = np.random.choice([0,1])
    vizinhancas = [move, double_move, block_change]
    args = (equipDB, MPDB, Pij)
    epsilon = epsilons[num_func_obj]
    e = epsilon()

    f_obj_Pe = make_objf_Pe(e, num_func_obj=num_func_obj)
    resultado = GVNS(f_obj_Pe, gerar_solucao_inicial, vizinhancas, busca_local, *args, max_iter=1000, return_curve=True)
    return resultado, num_func_obj, epsilon

def executar_uma_repeticao_Pe():
    with ProcessPoolExecutor(max_workers=10) as executor:
        resultados_com_epsilon = list(executor.map(epsilon_restrito, range(it)))
    return resultados_com_epsilon


# Número de iterações a ser executado (por exemplo, 32 iterações)
it = 20 #intancias paralelas
repeticoes = 5

if __name__ == "__main__":
    todas_as_frentes_pareto_Pw = []
    todas_as_frentes_pareto_Pe = []

    for rep in range(repeticoes):
        print(f"\n=== Repetição {rep+1}/{repeticoes} ===")
        resultados_com_pesos = executar_uma_repeticao_Pw()
        resultados_com_epislons = executar_uma_repeticao_Pe()
        args = (equipDB, MPDB, Pij)

        # Separando os resultados e os Ws
        resultados_Pw = [r[0] for r in resultados_com_pesos]
        pesos_W = [r[1] for r in resultados_com_pesos]

        resultados_Pe = [r[0] for r in resultados_com_epislons]
        num_func_obj_Pe = [r[1] for r in resultados_com_epislons]
        epsilons = [r[2] for r in resultados_com_epislons]


        all_sol=[]
       
        custos_f1_Pw = np.array([f1(sol, *args) for sol, _ in resultados_Pw])
        custos_f2_Pw = np.array([f2(sol, *args) for sol, _ in resultados_Pw])

        custos_f1_Pe = np.array([f1(sol, *args) for sol, _ in resultados_Pe])
        custos_f2_Pe = np.array([f2(sol, *args) for sol, _ in resultados_Pe])
       

        pareto_Pw = []
        for (sol, hist), W_local in zip(resultados_Pw, pesos_W):
            all_sol.append(sol)
            custo = f1(sol, *args)
            falha = f2(sol, *args)
            pareto_Pw.append((custo, falha))

        todas_as_frentes_pareto_Pw.append(np.array(pareto_Pw))

        pareto_Pe = []
        for (sol, hist), epsilon in zip(resultados_Pe, epsilons):
            all_sol.append(sol)
            custo = f1(sol, *args)
            falha = f2(sol, *args)
            pareto_Pe.append((custo, falha))

        todas_as_frentes_pareto_Pe.append(np.array(pareto_Pe))

    # Plotando todas as fronteiras de Pareto Pw
    plt.figure()
    cores = ['blue', 'green', 'red', 'orange', 'purple']
    for i, fronteira in enumerate(todas_as_frentes_pareto_Pw):
        fronteira_nd = filtrar_pareto_nao_dominado(fronteira)
        plt.scatter(fronteira[:, 0], fronteira[:, 1], label=f"Repetição {i+1}", alpha=0.6, color=cores[i % len(cores)])

    plt.title("Fronteiras de Pareto com Pw - 5 Repetições")
    plt.xlabel("f1: Custo de Manutenção")
    plt.ylabel("f2: Custo Esperado de Falha")
    plt.grid(True)
    plt.legend()
    plt.show()


    # Plotando todas as fronteiras de Pareto Pe
    plt.figure()
    cores = ['blue', 'green', 'red', 'orange', 'purple']
    for i, fronteira in enumerate(todas_as_frentes_pareto_Pe):
        fronteira_nd = filtrar_pareto_nao_dominado(fronteira)
        plt.scatter(fronteira[:, 0], fronteira[:, 1], label=f"Repetição {i+1}", alpha=0.6, color=cores[i % len(cores)])

    plt.title("Fronteiras de Pareto com Pe - 5 Repetições")
    plt.xlabel("f1: Custo de Manutenção")
    plt.ylabel("f2: Custo Esperado de Falha")
    plt.grid(True)
    plt.legend()
    plt.show()


    df=pd.DataFrame(all_sol)
    df.to_csv('Result.csv', index=False, header=False, sep=';')