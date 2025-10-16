
/* kmeans_1d_omp.c
 * K-means 1D (C99), implementação paralela OpenMP:
 * - Assignment step: Paralelizado com 'reduction(+:sse)'.
 * - Update step (Opção A): Acumuladores locais por thread, seguidos por redução serial.
 *
 * Compilar: gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
 * Uso:      export OMP_NUM_THREADS=4
 *          ./kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Biblioteca OpenMP

/* ---------- util CSV 1D: cada linha tem 1 número (Inalteradas) ---------- */

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
    double *A = (double*)malloc((size_t)R * sizeof(double));
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

        /* aceita vírgula/ponto-e-vírgula/espaco/tab, pega o primeiro token numérico */
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

/* ---------- k-means 1D (Paralelizado com OpenMP) ---------- */

/* assignment: para cada X[i], encontra c com menor (X[i]-C[c])^2
 * Paralelizado com reduction(+:sse)
 */
static double assignment_step_1d_omp(const double *X, const double *C, int *assign, int N, int K){
    double sse = 0.0;

    // Paraleliza o laço sobre N pontos. Usa 'reduction' para somar o SSE de forma segura.
    #pragma omp parallel for reduction(+:sse)
    for(int i=0;i<N;i++){
        int best = -1;
        double bestd = 1e300;

        // O laço interno sobre K centróides (c) permanece serial em cada thread.
        for(int c=0;c<K;c++){
            double diff = X[i] - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

/* update: média dos pontos de cada cluster (1D)
 * Opção A: Acumuladores locais por thread, seguidos por redução serial.
 */
static void update_step_1d_omp(const double *X, double *C, const int *assign, int N, int K){

    int max_threads = omp_get_max_threads();

    // Acumuladores locais por thread (t) e por cluster (K).
    // Array de tamanho: max_threads * K
    double *sum_thread = (double*)calloc((size_t)max_threads * K, sizeof(double));
    int *cnt_thread = (int*)calloc((size_t)max_threads * K, sizeof(int));
    if(!sum_thread || !cnt_thread){ fprintf(stderr,"Sem memoria no update OpenMP\n"); exit(1); }

    // PARALELIZAÇÃO: Acúmulo local
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        int offset = t * K;

        // Cada thread itera sobre seu bloco de N pontos e acumula nos seus arrays locais.
        #pragma omp for
        for(int i=0;i<N;i++){
            int a = assign[i];
            cnt_thread[offset + a] += 1;
            sum_thread[offset + a] += X[i];
        }
    }

    // REDUÇÃO (Serial): Combina os resultados das threads.
    // Acumuladores globais temporários para a soma total (sum e cnt).
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria para reducao\n"); exit(1); }

    for(int t=0; t<max_threads; t++){
        int offset = t * K;
        for(int c=0; c<K; c++){
            sum[c] += sum_thread[offset + c];
            cnt[c] += cnt_thread[offset + c];
        }
    }

    // UPDATE: Recálculo dos centróides (Serial, igual ao naive)
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else C[c] = X[0]; /* cluster vazio recebe o primeiro ponto */
    }

    free(sum_thread); free(cnt_thread);
    free(sum); free(cnt);
}

/* kmeans_1d: O loop principal do algoritmo (Inalterado, apenas chama as versões OMP) */
static void kmeans_1d_omp(const double *X, double *C, int *assign,
                         int N, int K, int max_iter, double eps,
                         int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d_omp(X, C, assign, N, K);
        /* parada por variação relativa do SSE */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }
        update_step_1d_omp(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main (Inalterada, apenas muda o nome do algoritmo nos logs) ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    clock_t t0 = clock();
    int iters = 0; double sse = 0.0;
    // Chama a função principal OMP
    kmeans_1d_omp(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D (OpenMP)\n"); // Mudar a saída para indicar a versão
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}
