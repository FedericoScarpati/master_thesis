#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)
#define sign(x) ((x) > 0 ? 1 : -1)
#define max(a,b) ((a) > (b) ? (a) : (b))

#define Q 5

struct vector {
  int n[Q];
} zero, sum, group[Q], *perm;

int N, NoverQ, M, *graph, **neigh, *deg, *color, numUnsat, fact[Q+1];

/* variabili globali per il generatore random */
unsigned myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

unsigned randForInit(void) {
  unsigned long long y;
  
  y = myrand * 16807LL;
  myrand = (y & 0x7fffffff) + (y >> 31);
  if (myrand & 0x80000000) {
    myrand = (myrand & 0x7fffffff) + 1;
  }
  return myrand;
}

void initRandom(void) {
  int i;
  
  ip = 128;    
  ip1 = ip - 24;    
  ip2 = ip - 55;    
  ip3 = ip - 61;
  
  for (i = ip3; i < ip; i++) {
    ira[i] = randForInit();
  }
}

float gaussRan(void) {
  static int iset = 0;
  static float gset;
  float fac, rsq, v1, v2;
  
  if (iset == 0) {
    do {
      v1 = 2.0 * FRANDOM - 1.0;
      v2 = 2.0 * FRANDOM - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return v2 * fac;
  } else {
    iset = 0;
    return gset;
  }
}

void error(char *string) {
  fprintf(stderr, "ERROR: %s\n", string);
  exit(EXIT_FAILURE);
}

void allocateMem(void) {
  int i;

  fact[0] = 1;
  for ( i = 0; i < Q; i++) {
    zero.n[i] = 0;
    fact[i+1] = (i+1) * fact[i];
  }
  graph = (int*)calloc(2*M, sizeof(int));
  deg = (int*)calloc(N, sizeof(int));
  neigh = (int**)calloc(N, sizeof(int*));
  color = (int*)calloc(N, sizeof(int));
  perm = (struct vector*)calloc(fact[Q], sizeof(struct vector));
}

void printPerm(void) {
  int i, j;
  
  for (i = 0; i < fact[Q]; i++) {
    for (j = 0; j < Q; j++)
      printf("%i", perm[i].n[j]);
    printf("\n");
  }
  printf("\n");
}

struct vector rotate(struct vector input, int modulo) {
  struct vector output;
  int i;

  for (i = 0; i < modulo; i++)
    output.n[i] = (input.n[i] + 1) % modulo;
  return output;
}

void initPerm(int max) {
  int i;
  
  if (max == 1)
    perm[0].n[0] = 0;
  else {
    initPerm(max-1);
    for (i = 0; i < fact[max-1]; i++)
      perm[i].n[max-1] = max-1;
    for (; i < fact[max]; i++)
      perm[i] = rotate(perm[i-fact[max-1]], max);
  }
}

void makeGraph(void) {
  int i, var1, var2;
  FILE *graphFile;

  graphFile = fopen("generated_graphs.txt", "a");
  if (graphFile == NULL) {
    fprintf(stderr, "Impossibile aprire il file dei grafi.\n");
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < N; i++)
    deg[i] = 0;
  for (i = 0; i < M; i++) {
    var1 = (int)(FRANDOM * N);
    do {
      var2 = (int)(FRANDOM * N);
    } while ((int)(var1/NoverQ) == (int)(var2/NoverQ));
    graph[2*i] = var1;
    graph[2*i+1] = var2;
    deg[var1]++;
    deg[var2]++;
    fprintf(graphFile, "%i %i ", var1, var2);
  }
    fprintf(graphFile, "\n");


  for (i = 0; i < N; i++) {
    neigh[i] = (int*)calloc(deg[i], sizeof(int));
    deg[i] = 0;
  }
  for (i = 0; i < M; i++) {
    var1 = graph[2*i];
    var2 = graph[2*i+1];
    neigh[var1][deg[var1]++] = var2;
    neigh[var2][deg[var2]++] = var1;
  }
  fclose(graphFile);
}

void initColors(void) {
  int i;
  
  for (i = 0; i < N; i++)
    color[i] = (int)(FRANDOM * Q);
}

void oneMCS(double temperature) {
  int i, j, newColor, deltaUnsat;

  for (i = 0; i < N; i++) {
    do {
      newColor = (int)(FRANDOM * Q);
    } while (newColor == color[i]);
    sum = zero;
    for (j = 0; j < deg[i]; j++)
      sum.n[color[neigh[i][j]]]++;
    deltaUnsat = sum.n[newColor] - sum.n[color[i]];
    if (deltaUnsat <= 0 || deltaUnsat < -temperature * log(FRANDOM)) {
      color[i] = newColor;
      numUnsat += deltaUnsat;
    }
  }
}

void quenchT0(int maxIter) {
  int i, j, newColor, deltaUnsat, t, changes;

  t = 0;
  do {
    changes = 0;
    for (i = 0; i < N; i++) {
      do {
	newColor = (int)(FRANDOM * Q);
      } while (newColor == color[i]);
      sum = zero;
      for (j = 0; j < deg[i]; j++)
	sum.n[color[neigh[i][j]]]++;
      deltaUnsat = sum.n[newColor] - sum.n[color[i]];
      if (deltaUnsat <= 0) {
	color[i] = newColor;
	numUnsat += deltaUnsat;
	changes++;
      }
    }
    t++;
  } while (changes && t < maxIter);
}

int computeUnsat(void) {
  int i, res = 0;

  for (i = 0; i < M; i++)
    res += (color[graph[2*i]] == color[graph[2*i+1]]);
  return res;
}

double overlapPlanted(void) {
  int i, j, overlap, maxOver=0;

  for (i = 0; i < Q; i++)
    group[i] = zero;
  for (i = 0; i < N; i++)
    group[(int)(i/NoverQ)].n[color[i]]++;
  for (i = 0; i < fact[Q]; i++) {
    overlap = 0;
    for (j = 0; j < Q; j++)
      overlap += group[j].n[perm[i].n[j]];
    if (overlap > maxOver) maxOver = overlap;
  }
  return (double)(Q*maxOver-N)/(Q-1)/N;
}

void freeMem(void) {
  int i;
  for (i = 0; i < N; i++)
    free(neigh[i]);
}


int main(int argc, char *argv[]) {
  int i, nIter, measTime;
  double c, temp, startTemp;
  FILE *devran = fopen("/dev/urandom","r");
  FILE *file;

  fread(&myrand, 4, 1, devran);
  fclose(devran);

  if (argc != 5) {
    fprintf(stderr, "usage: %s N c <startTemp> <nIter>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

    // Apri il file in modalità append
  file = fopen("numUnsat_log.txt", "a");
  if (file == NULL) {
    fprintf(stderr, "Impossibile aprire il file di log.\n");
    return EXIT_FAILURE;
  }

  N = atoi(argv[1]);
  c = atof(argv[2]);
  startTemp = atof(argv[3]);
  nIter = atoi(argv[4]);
  if (Q * (int)(N/Q) != N) error("Q must divide N");
  NoverQ = (int)(N/Q);
  M = (int)(0.5 * c * N + 0.5);

  
  measTime = (int)(nIter / 1000);
  printf("# Q = %i   N = %i   M = %i   c = %f   startTemp = %f   nIter = %i   seed = %u\n",
	 Q, N, M, c, startTemp, nIter, myrand);
  printf("# 1:T  2:numUnsat  3:overlapPlanted\n");
  initRandom();
  allocateMem();
  initPerm(Q);
  //printPerm();


  makeGraph();
  initColors();
 
  numUnsat = computeUnsat();
  temp = startTemp;
  for (i = nIter; i > 0 && numUnsat; i--) {
    temp = i * startTemp / nIter;
    oneMCS(temp);
    if (numUnsat != computeUnsat()) error("in unsat");
    printf("%i %.3f %i %g\n", i, temp, numUnsat, overlapPlanted());

    // stampa energia ogni 10 step
    if (i % 10 == 0){
        fprintf(file, "%i ", numUnsat);
    }

    // stampa energia finale
    if (numUnsat == 0){
      fprintf(file, "0 ");
    }
  }
  printf("%.3f %i %g\n", temp, numUnsat, overlapPlanted());
  if (numUnsat) {
    quenchT0(10);
    printf("0.0 %i %g\n", numUnsat, overlapPlanted());
  }
  printf("\n");
  fprintf(file, "\n");
  fflush(stdout);
  fclose(file);

 
  // Log file N, c, numUnsat, numIter
 
  file = fopen("simann_log.txt", "a");
  if (file == NULL) {
    fprintf(stderr, "Impossibile aprire il file di log.\n");
    return EXIT_FAILURE;
  }
  // Verifica se il file è vuoto. Se è vuoto, scrive l'header.
  fseek(file, 0, SEEK_END);
  if (ftell(file) == 0) {
    fprintf(file, "N,c,numUnsat,nIter,overlap\n"); // Header del CSV
  }
  fseek(file, 0, SEEK_END);
  fprintf(file, "%i,%f,%i,%i,%f\n", N, c, numUnsat, nIter-i, overlapPlanted());
  fclose(file);

  freeMem();
  
  return EXIT_SUCCESS;
}