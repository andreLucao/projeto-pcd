/* kmeans_1d_mpi.c
 * K-means 1D com MPI (memória distribuída)
 * 
 * Estratégia:
 * - Processo 0 (root) lê os dados e distribui entre os processos
 * - Cada processo trabalha com um bloco local de pontos
 * - Centróides são broadcast para todos os processos
 * - Assignment é feito localmente
 * - Update usa MPI_Allreduce para somar counts e sums globalmente
 * 
 * Compilar: mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
 * Executar: mpirun -np 4 ./kmeans_1d_mpi dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ---------- Funções auxiliares de leitura CSV ---------- */

static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        exit(1);
    }
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out) {
    int R = count_rows(path);
    if (R <= 0) {
        fprintf(stderr, "Arquivo vazio: %s\n", path);
        exit(1);
    }
    double *A = (double *)malloc((size_t)R * sizeof(double));
    if (!A) {
        fprintf(stderr, "Sem memoria para %d linhas\n", R);
        exit(1);
    }
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        free(A);
        exit(1);
    }
    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (only_ws) continue;
        
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok) {
            fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path);
            free(A);
            exit(1);
        }
        A[r] = atof(tok);
        r++;
        if (r > R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int c = 0; c < K; c++)
        fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- K-means MPI ---------- */

/* Assignment local: cada processo calcula para seus pontos */
static double assignment_step_local(const double *X_local, const double *C,
                                     int *assign_local, int N_local, int K) {
    double sse_local = 0.0;
    for (int i = 0; i < N_local; i++) {
        int best = -1;
        double bestd = 1e300;
        for (int c = 0; c < K; c++) {
            double diff = X_local[i] - C[c];
            double d = diff * diff;
            if (d < bestd) {
                bestd = d;
                best = c;
            }
        }
        assign_local[i] = best;
        sse_local += bestd;
    }
    return sse_local;
}

/* Update: calcula somas e contagens locais, depois reduz globalmente */
static void update_step_mpi(const double *X_local, double *C,
                             const int *assign_local, int N_local, int K,
                             const double *X_full, int N_full) {
    double *sum_local = (double *)calloc((size_t)K, sizeof(double));
    int *cnt_local = (int *)calloc((size_t)K, sizeof(int));
    double *sum_global = (double *)calloc((size_t)K, sizeof(double));
    int *cnt_global = (int *)calloc((size_t)K, sizeof(int));
    
    if (!sum_local || !cnt_local || !sum_global || !cnt_global) {
        fprintf(stderr, "Sem memoria no update\n");
        exit(1);
    }
    
    /* Acumula localmente */
    for (int i = 0; i < N_local; i++) {
        int a = assign_local[i];
        cnt_local[a] += 1;
        sum_local[a] += X_local[i];
    }
    
    /* Redução global com Allreduce */
    MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    /* Atualiza centróides */
    for (int c = 0; c < K; c++) {
        if (cnt_global[c] > 0) {
            C[c] = sum_global[c] / (double)cnt_global[c];
        } else {
            /* Cluster vazio: estratégia simples, usa X[0] */
            if (X_full != NULL) {
                C[c] = X_full[0];
            }
        }
    }
    
    free(sum_local);
    free(cnt_local);
    free(sum_global);
    free(cnt_global);
}

/* ---------- Main ---------- */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 3) {
        if (rank == 0) {
            printf("Uso: mpirun -np <P> %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc > 5) ? argv[5] : NULL;
    const char *outCentroid = (argc > 6) ? argv[6] : NULL;
    
    int N = 0, K = 0;
    double *X_full = NULL;
    double *C = NULL;
    int *assign_full = NULL;
    
    /* Processo 0 lê os dados */
    if (rank == 0) {
        X_full = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
        assign_full = (int *)malloc((size_t)N * sizeof(int));
        if (!assign_full) {
            fprintf(stderr, "Sem memoria para assign\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    /* Broadcast N e K para todos os processos */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Aloca centróides em todos os processos */
    if (rank != 0) {
        C = (double *)malloc((size_t)K * sizeof(double));
        if (!C) {
            fprintf(stderr, "Processo %d: sem memoria para C\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    /* Calcula distribuição dos dados */
    int *sendcounts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    
    int base = N / size;
    int remainder = N % size;
    
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }
    
    int N_local = sendcounts[rank];
    double *X_local = (double *)malloc((size_t)N_local * sizeof(double));
    int *assign_local = (int *)malloc((size_t)N_local * sizeof(int));
    
    if (!X_local || !assign_local) {
        fprintf(stderr, "Processo %d: sem memoria para dados locais\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Distribui os dados X entre os processos */
    MPI_Scatterv(X_full, sendcounts, displs, MPI_DOUBLE,
                 X_local, N_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* Inicia timer */
    double t_start = MPI_Wtime();
    
    /* Iterações do K-means */
    double prev_sse = 1e300;
    double sse_local, sse_global = 0.0;
    int it;
    
    for (it = 0; it < max_iter; it++) {
        /* Broadcast dos centróides atuais */
        MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        /* Assignment local */
        sse_local = assignment_step_local(X_local, C, assign_local, N_local, K);
        
        /* Reduz SSE global */
        MPI_Allreduce(&sse_local, &sse_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        /* Verifica convergência */
        double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps) {
            it++;
            break;
        }
        
        /* Update dos centróides */
        update_step_mpi(X_local, C, assign_local, N_local, K, X_full, N);
        
        prev_sse = sse_global;
    }
    
    /* Finaliza timer */
    double t_end = MPI_Wtime();
    double elapsed = (t_end - t_start) * 1000.0; // em ms
    
    /* Coleta assignments de volta para o processo 0 */
    MPI_Gatherv(assign_local, N_local, MPI_INT,
                assign_full, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    
    /* Processo 0 imprime resultados e salva arquivos */
    if (rank == 0) {
        printf("K-means 1D (MPI)\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("Processos MPI: %d\n", size);
        printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", it, sse_global, elapsed);
        
        write_assign_csv(outAssign, assign_full, N);
        write_centroids_csv(outCentroid, C, K);
        
        free(X_full);
        free(assign_full);
    }
    
    /* Limpeza */
    free(X_local);
    free(assign_local);
    free(C);
    free(sendcounts);
    free(displs);
    
    MPI_Finalize();
    return 0;
}