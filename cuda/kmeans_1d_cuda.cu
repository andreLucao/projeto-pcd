/* kmeans_1d_cuda.cu
   K-means 1D com CUDA (Etapa 2):
   - Kernel de assignment na GPU (1 thread por ponto)
   - Update no host (opção A: mais simples e estável)
   - Centróides em memória constante (otimização)
   - Mesma interface/IO do código sequencial

   Compilar: nvcc -O2 kmeans_1d_cuda.cu -o kmeans_1d_cuda
   Uso:      ./kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// Memória constante para centróides (otimização)
#define MAX_K 64
_constant_ double d_C_const[MAX_K];

/* ---------- Macros de verificação CUDA ---------- */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", _FILE, __LINE_, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double A = (double)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- CUDA Kernels ---------- */

// Kernel de assignment: cada thread processa um ponto
_global_ void assignment_kernel(const double *d_X, const int *d_use_const, 
                                  const double *d_C_global, int *d_assign, 
                                  double *d_errors, int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double x = d_X[i];
    int best = -1;
    double bestd = 1e300;

    // Usar memória constante se K <= MAX_K, senão global
    for(int c = 0; c < K; c++){
        double centroid = (*d_use_const) ? d_C_const[c] : d_C_global[c];
        double diff = x - centroid;
        double d = diff * diff;
        if(d < bestd){ 
            bestd = d; 
            best = c; 
        }
    }

    d_assign[i] = best;
    d_errors[i] = bestd;
}

// Redução paralela para calcular SSE total (redução em blocos)
_global_ void reduce_sse_kernel(const double *d_errors, double *d_block_sums, int N)
{
    extern _shared_ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Carregar dados para memória compartilhada
    sdata[tid] = (i < N) ? d_errors[i] : 0.0;
    __syncthreads();
    
    // Redução na memória compartilhada
    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 escreve resultado do bloco
    if(tid == 0){
        d_block_sums[blockIdx.x] = sdata[0];
    }
}

/* ---------- k-means 1D com CUDA ---------- */

// Update no host (CPU) - opção A: mais simples e estável
static void update_step_1d_host(const double *X, double *C, const int *assign, int N, int K){
    double sum = (double)calloc((size_t)K, sizeof(double));
    int cnt = (int)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    for(int i=0; i<N; i++){
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }

    for(int c=0; c<K; c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0]; // cluster vazio recebe o primeiro ponto
    }

    free(sum); free(cnt);
}

static void kmeans_1d_cuda(const double *X, double *C, int *assign,
                           int N, int K, int max_iter, double eps,
                           int *iters_out, double *sse_out, int block_size,
                           double *time_h2d, double *time_kernel, double *time_d2h)
{
    // Alocação de memória no device
    double *d_X, *d_C_global, *d_errors;
    int *d_assign, *d_use_const;
    
    CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C_global, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_assign, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_errors, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_use_const, sizeof(int)));

    // Configuração de grid e blocos
    int grid_size = (N + block_size - 1) / block_size;
    
    // Para redução de SSE
    double *d_block_sums;
    int num_blocks_reduce = grid_size;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks_reduce * sizeof(double)));
    double h_block_sums = (double)malloc(num_blocks_reduce * sizeof(double));

    // Eventos CUDA para medir tempo
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    *time_h2d = 0.0;
    *time_kernel = 0.0;
    *time_d2h = 0.0;

    // Copiar X para device (uma vez só)
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    *time_h2d += ms;

    // Decidir se usa memória constante
    int use_const = (K <= MAX_K) ? 1 : 0;
    CUDA_CHECK(cudaMemcpy(d_use_const, &use_const, sizeof(int), cudaMemcpyHostToDevice));

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;

    for(it = 0; it < max_iter; it++){
        // Copiar centróides para device
        CUDA_CHECK(cudaEventRecord(start));
        if(use_const){
            CUDA_CHECK(cudaMemcpyToSymbol(d_C_const, C, K * sizeof(double)));
        } else {
            CUDA_CHECK(cudaMemcpy(d_C_global, C, K * sizeof(double), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        *time_h2d += ms;

        // Assignment kernel
        CUDA_CHECK(cudaEventRecord(start));
        assignment_kernel<<<grid_size, block_size>>>(d_X, d_use_const, d_C_global, 
                                                     d_assign, d_errors, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        *time_kernel += ms;

        // Redução para calcular SSE
        CUDA_CHECK(cudaEventRecord(start));
        int shared_mem_size = block_size * sizeof(double);
        reduce_sse_kernel<<<num_blocks_reduce, block_size, shared_mem_size>>>(d_errors, d_block_sums, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        *time_kernel += ms;

        // Copiar resultados parciais e finalizar redução no host
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, num_blocks_reduce * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        *time_d2h += ms;

        sse = 0.0;
        for(int b = 0; b < num_blocks_reduce; b++){
            sse += h_block_sums[b];
        }

        // Verificar convergência
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }

        // Copiar assign para host para fazer update
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        *time_d2h += ms;

        // Update no host
        update_step_1d_host(X, C, assign, N, K);
        prev_sse = sse;
    }

    // Copiar assignment final
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    *time_d2h += ms;

    *iters_out = it;
    *sse_out = sse;

    // Limpeza
    free(h_block_sums);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C_global));
    CUDA_CHECK(cudaFree(d_assign));
    CUDA_CHECK(cudaFree(d_errors));
    CUDA_CHECK(cudaFree(d_use_const));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [block_size=256] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    int block_size = (argc>5)? atoi(argv[5]) : 256;
    const char *outAssign   = (argc>6)? argv[6] : NULL;
    const char *outCentroid = (argc>7)? argv[7] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    if(block_size <= 0 || block_size > 1024){
        fprintf(stderr,"block_size deve estar entre 1 e 1024\n");
        return 1;
    }

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int assign = (int)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    // Informações da GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    double time_h2d, time_kernel, time_d2h;
    cudaEvent_t start_total, stop_total;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    
    CUDA_CHECK(cudaEventRecord(start_total));
    
    int iters = 0; 
    double sse = 0.0;
    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, &iters, &sse, block_size,
                   &time_h2d, &time_kernel, &time_d2h);
    
    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));
    
    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_total, stop_total));

    // MESMO formato de saída + informações CUDA
    printf("K-means 1D (CUDA)\n");
    printf("GPU: %s\n", prop.name);
    printf("N=%d K=%d max_iter=%d eps=%g block_size=%d\n", N, K, max_iter, eps, block_size);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, total_ms);
    printf("  H2D: %.1f ms | Kernel: %.1f ms | D2H: %.1f ms\n", time_h2d, time_kernel, time_d2h);
    printf("Throughput: %.2f Mpontos/s\n", (N * iters) / (total_ms * 1000.0));

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    
    return 0;
}