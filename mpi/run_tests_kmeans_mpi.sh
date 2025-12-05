#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------
# Configurações
# ---------------------------------------
EXEC=./kmeans_1d_mpi
HOSTFILE=mpi_hosts

# Números de processos MPI a testar
NP_LIST=("1" "2" "4")

# Datasets e arquivos de centróides
DATASETS=("dados_pequeno.csv" "dados_medio.csv" "dados_grande.csv")
CENTROIDS=("centroides_iniciais_k4.csv" "centroides_iniciais_k8.csv" "centroides_iniciais_k16.csv")

MAX_ITER=50
EPS=1e-4

RESULTS_CSV="resultados_kmeans_mpi.csv"

# ---------------------------------------
# Cabeçalho do arquivo de resultados
# ---------------------------------------
echo "dataset,K,np,iteracoes,sse_final,tempo_ms" > "$RESULTS_CSV"

# ---------------------------------------
# Loop de testes
# ---------------------------------------
for data in "${DATASETS[@]}"; do
  base_data=$(basename "$data" .csv)

  for cent in "${CENTROIDS[@]}"; do
    base_cent=$(basename "$cent" .csv)

    # extrai K do nome do arquivo (k4, k8, k16 -> 4, 8, 16)
    K=$(echo "$base_cent" | sed -E 's/.*k([0-9]+).*/\1/')

    for np in "${NP_LIST[@]}"; do
      echo "============================================================"
      echo "Dataset: $data | K=$K | np=$np"
      echo "------------------------------------------------------------"

      # nomes de saída
      out_assign="assign_${base_data}_k${K}_np${np}.csv"
      out_cent="centroides_${base_data}_k${K}_np${np}.csv"
      log_file="log_${base_data}_k${K}_np${np}.txt"

      # executa o programa com mpirun (sem DISPLAY p/ não encher de aviso)
      env -u DISPLAY mpirun \
  --mca btl_tcp_if_include enp0s8 \
  --mca oob_tcp_if_include enp0s8 \
  -np "$np" -hostfile "$HOSTFILE" \
  "$EXEC" "$data" "$cent" "$MAX_ITER" "$EPS" \
  "$out_assign" "$out_cent" | tee "$log_file"

      # pega a linha com Iterações / SSE / Tempo
      summary=$(grep "Iterações:" "$log_file" || true)

      # se não encontrou (erro), registra vazio
      if [[ -z "$summary" ]]; then
        echo "  [AVISO] Não foi possível extrair resumo da execução."
        echo "${base_data},${K},${np},,,," >> "$RESULTS_CSV"
      else
        # ex.: Iterações: 3 | SSE final: 9965.693552 | Tempo: 11.1 ms
        iter=$(echo "$summary"  | awk -F'[:|]' '{gsub(/ /,"",$2); print $2}')
        sse=$(echo "$summary"   | awk -F'[:|]' '{gsub(/ /,"",$3); print $3}')
        tempo=$(echo "$summary" | awk -F'[:|]' '{gsub(/ ms/,"",$4); gsub(/ /,"",$4); print $4}')

        echo "  -> Iterações = $iter | SSE = $sse | Tempo = ${tempo} ms"

        # grava no CSV consolidado
        echo "${base_data},${K},${np},${iter},${sse},${tempo}" >> "$RESULTS_CSV"
      fi

      echo
    done
  done
done

echo "============================================================"
echo "Testes concluídos. Resultados consolidados em: $RESULTS_CSV"
