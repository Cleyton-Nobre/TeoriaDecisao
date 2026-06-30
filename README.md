# Política Ótima de Manutenção — Trabalho Computacional (Teoria da Decisão)

Trabalho que aplica conceitos de **Teoria da Decisão** e **otimização multiobjetivo** para definir a melhor política de manutenção preventiva para **500 equipamentos**, equilibrando custo de manutenção e custo esperado de falha.

**Autores:** Cleyton Nobre, Thales Tenebra, Thiago Fraga (UFMG)

## Problema

Cada equipamento `i` deve ser alocado a um de três planos de manutenção `j` (sem manutenção, plano médio, plano detalhado), com base em:
- Idade atual do equipamento
- Cluster ao qual pertence (define os parâmetros da distribuição de Weibull)
- Custo de falha associado

São otimizadas duas funções objetivo conflitantes:
- **f1** — custo total de manutenção
- **f2** — custo esperado de falha (probabilidade de falha × custo de falha, via distribuição de Weibull)

## Metodologia

1. **Dados:** `EquipDB.csv`, `ClusterDB.csv`, `MPDB.csv`
2. **Modelagem probabilística:** probabilidade de falha condicional via CDF de Weibull
3. **Solução inicial:** equipamentos ordenados por criticidade (custo de falha × probabilidade × tempo desde a última falha)
4. **Metaheurística:** GVNS (General Variable Neighborhood Search), com vizinhanças `move`, `double_move` e `block_change`
5. **Abordagens multiobjetivo:**
   - **Soma ponderada (Pw):** combina f1 e f2 normalizadas com peso aleatório `w`
   - **ε-restrito (Pε):** minimiza uma função tratando a outra como restrição penalizada
6. **Avaliação da fronteira de Pareto:** indicador de **hipervolume**
7. **Critérios adicionais de decisão:**
   - f3 — desbalanceamento do MTTF entre planos
   - f4 — entropia da diversidade de planos por cluster
8. **Escolha final da solução:** métodos multicritério **PROMETHEE II** e **ELECTRE I**

## Principais Resultados

**Otimização mono-objetivo (GVNS, 5 execuções):**

| Função | Mínimo | Máximo | Desvio Padrão |
|---|---|---|---|
| f1 (custo de manutenção) | \$28,0 | \$40,0 | \$4,41 |
| f2 (custo esperado de falha) | \$1.061,63 | \$1.063,60 | \$0,65 |

**Fronteira de Pareto (hipervolume):**
- Soma ponderada (Pw): **0,6177** (≈61,8% de cobertura)
- ε-restrito (Pε): **0,6384** (≈63,8% de cobertura)

Ambos os métodos produziram fronteiras consistentes e robustas entre repetições; Pε obteve cobertura ligeiramente superior, mas o desempenho é considerado equivalente dado o componente aleatório de ambas as abordagens.

**Escolha final da solução:**

| Critério | PROMETHEE II | ELECTRE I |
|---|---|---|
| f1 | 724 | 708 |
| f2 | 1.165,90 | 1.170,24 |
| f3 | 1.221,37 | 1.191,57 |
| f4 | 1,286 | 1,308 |
| Custo combinado (f1+f2) | \$1.889,90 | **\$1.878,24** |

➡️ A solução escolhida como política final de manutenção foi a obtida pelo **método ELECTRE I**, por apresentar o menor custo combinado.

## Conclusão

As abordagens de soma ponderada e ε-restrito geraram soluções estáveis e bem distribuídas próximas à fronteira de Pareto teórica. A incorporação de critérios adicionais (entropia e balanceamento de MTTF) enriqueceu a análise, e os métodos multicritério permitiram uma escolha sistemática e objetiva da melhor política de manutenção entre 500 equipamentos.

---

> **Nota:** as imagens de convergência (`conv-f1-1000.png`, `conv-f2-1000.png`) e das fronteiras de Pareto (`fronteiraParetoPw.jpeg`, `pareto_pe.png`, etc.) referenciadas no documento original não estavam disponíveis no momento da geração deste README. Envie os arquivos para que eu possa incluí-los.