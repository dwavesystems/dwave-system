/*
Copyright 2016 D-Wave Systems Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "dwave_sapi.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

/* I make this with
gcc solve_biclique.c -o solve_biclique libdwave_sapi.so -lm
*/


void handle_error(sapi_Code error, const char *function, const char *more_error) {
    if(error == SAPI_OK) {
        return;
    }
    printf("Error encountered at (%s) calling dwave_sapi function: ", function);
    switch( (int)error ) {
        case SAPI_ERR_INVALID_PARAMETER:
            printf("invalid parameter"); break;
        case SAPI_ERR_SOLVE_FAILED:
            printf("solve failed"); break;
        case SAPI_ERR_AUTHENTICATION:
            printf("authentication error"); break;
        case SAPI_ERR_NETWORK:
            printf("network error"); break;
        case SAPI_ERR_COMMUNICATION:
            printf("communication error"); break;
        case SAPI_ERR_ASYNC_NOT_DONE:
            printf("async not done"); break;
        case SAPI_ERR_PROBLEM_CANCELLED:
            printf("problem cancelled"); break;
        case SAPI_ERR_NO_INIT:
            printf("not initialized"); break;
        case SAPI_ERR_OUT_OF_MEMORY:
            printf("out of memory"); break;
        default:
            printf("unknown error, %d", (int) error);
    }
    printf("\n");
    if(more_error != NULL) {
        printf("%s\n", more_error);
    }
}


int main() {
    int i,j;

    /* always initialize first */
    sapi_Code retval;
    retval = sapi_globalInit();
    if (retval != SAPI_OK) {
        handle_error(retval, "globalInit", NULL);
        goto CLEANUP0;
    }

    /* grab a local connection */
    sapi_Connection *connection = sapi_localConnection();

    /* or, connect to a remote solver.  You'll need to put in your authentication information */
    /*
    char *sapi_url = NULL;
    char *sapi_token = NULL;
    char *proxy_url = NULL;
    code = sapi_remoteConnection(sapi_url, sapi_token, proxy_url, &connection, err_msg);
    if (code != SAPI_OK)
    {
        printf("%s\n", err_msg);
        goto CLEANUP0;
    }
    */

    /* connect to a local solver... you probably want to change this to use the hardware */
    char *solver_name = "c4-sw_sample";
    sapi_Solver *solver = sapi_getSolver(connection, solver_name);
    if (solver == NULL) {
        printf("Could not get the solver named '%s'.\n", solver_name);
        retval = -1;
        goto CLEANUP1;
    }

    /* grab the properties for this solver */
    const sapi_SolverProperties *solver_properties = sapi_getSolverProperties(solver);

    /* fetch the hardware adjaceny */
    sapi_Problem *hardware_adj;
    retval = sapi_getHardwareAdjacency(solver, &hardware_adj);
    if(retval != SAPI_OK) {
        handle_error(retval, "getHardwareAdjacency", NULL);
        goto CLEANUP2;
    }


    /* now let's load in the embedding from file.  first, we initialize the embedding object */
    sapi_Embeddings embedding;
    int N_q = embedding.len = solver_properties->quantum_solver->num_qubits;
    embedding.elements = (int *)malloc(N_q * sizeof(int));
    for(i=0;i<embedding.len;i++) {
        /* qubits labeled -1 aren't used in the embedding, some of these will be overwritten */
        embedding.elements[i] = -1;
    }

    /* now read in the file */
    FILE *embedding_file = fopen("embedded_biclique_largest","r");
    int qubit = -1;
    int n1 = 0;
    while(!feof(embedding_file) && fscanf(embedding_file, "%d", &qubit) && qubit != -1) {
        embedding.elements[qubit] = n1;
        if(feof(embedding_file) || fgetc(embedding_file) != ' ') {
            n1 ++;
        }
        qubit = -1;
    }
    int n2 = 0;
    while(!feof(embedding_file) && fscanf(embedding_file, "%d", &qubit) && qubit != -1) {
        embedding.elements[qubit] = n1+n2;
        if(feof(embedding_file) || fgetc(embedding_file) != ' ') {
            n2 ++;
        }
        qubit = -1;
    }

    fclose(embedding_file);

    /* generate the problem, start by initializing the problem object */
    sapi_Problem random_problem;
    random_problem.len = n1*n2 + n1+n2;
    random_problem.elements = (sapi_ProblemEntry *) malloc(random_problem.len * sizeof(sapi_ProblemEntry));

    /* set some random h biases */
    sapi_ProblemEntry *entry;
    for(i = 0; i < (n1+n2); i++) {
        entry = random_problem.elements + i;
        entry->i = entry->j = i;
        entry->value = 1 - (rand() * 2.0 / RAND_MAX);
    }

    /* set the J_{i,j} couplings */
    int offset = n1+n2;
    for(i = 0; i < n1; i++) {
        for(j=0; j < n2; j++) {
            entry = random_problem.elements + offset;
            entry->i = i;
            entry->j = n1+j;
            entry->value = 1 - (rand() * 2.0 / RAND_MAX);
            offset ++;
        }
    }

    /* now we've got the pieces in place to embed our problem */
    sapi_EmbedProblemResult *pre_embedded_problem;
    char err_msg[SAPI_ERROR_MESSAGE_MAX_SIZE];
    retval = sapi_embedProblem(&random_problem,  &embedding, hardware_adj, 0,0,NULL,
                                &pre_embedded_problem, err_msg);
    if(retval != SAPI_OK) {
        handle_error(retval, "embedProblem", err_msg);
        goto CLEANUP3;
    }

    /* in the python examples, we showcase using a couple of chain strengths.
       we'll just go with .25 here with hopes that we get some boken chains. */
    double chain_strength = .25; /* sqrt(n); */
    sapi_Problem embedded_problem;
    embedded_problem.len = pre_embedded_problem->problem.len + pre_embedded_problem->jc.len;
    embedded_problem.elements = (sapi_ProblemEntry *) malloc(embedded_problem.len * sizeof(sapi_ProblemEntry));
    memcpy(embedded_problem.elements, pre_embedded_problem->problem.elements,
           pre_embedded_problem->problem.len * sizeof(sapi_ProblemEntry));
    offset = pre_embedded_problem->problem.len;
    for(i=0;i < pre_embedded_problem->jc.len; i++) {
        entry = embedded_problem.elements + offset + i;
        entry->i = pre_embedded_problem->jc.elements[i].i;
        entry->j = pre_embedded_problem->jc.elements[i].j;
        entry->value = -chain_strength;
    }

    /* now we've actually got the embedded problem! let's solve it! */

    /* NOTE: if you're using a solver other than "c4-sw_sample" then you'll need to use a type
             other than sapi_SwSampleSolverParameters.  Please consult the documentation for
             sapi_solveIsing. */

    sapi_SwSampleSolverParameters params = SAPI_SW_SAMPLE_SOLVER_DEFAULT_PARAMETERS;
    params.num_reads = 1; /* this is an inefficient way to use the hardware! */
    sapi_IsingResult *samples;
    sapi_solveIsing(solver, &embedded_problem, (sapi_SolverParameters*)&params, &samples, err_msg);

    /* we expect the above to contain some broken chains, so let's patch that up. */
    /* there are a few strategies, the MINIMIZE_ENERGY strategy requires the most input from us
       (specifically, the original problem) so we show how to do that here. */
    /* first, make space to store the new solutions */
    int *new_solutions = (int *)malloc(samples->num_solutions * (n1+n2) * sizeof(int));
    size_t num_new_solutions;
    retval = sapi_unembedAnswer(samples->solutions, samples->solution_len, samples->num_solutions,
                                &embedding, SAPI_BROKEN_CHAINS_MINIMIZE_ENERGY, &random_problem,
                                new_solutions, &num_new_solutions, err_msg);
    if(retval != SAPI_OK) {
        handle_error(retval, "unembedSolution", err_msg);
        goto CLEANUP4;
    }
    /* Now, let's see what we got */

    printf("spin | chain spins:\n");
    for(i=0;i<(n1+n2);i++) {
        fflush(stdout);
        switch(new_solutions[i]) {
            case  1: printf("  +  | "); break;
            case -1: printf("  -  | "); break;
            default: printf("  ?  | "); break; /* this shouldn't happen */
        }
        for(j=0;j<N_q;j++) {
            if (embedding.elements[j] == i) {
                switch(samples->solutions[j]) {
                    case  1: printf("+"); break;
                    case -1: printf("-"); break;
                    default: printf("?"); break; /* this shouldn't happen */
                }
            }
        }
        printf("\n");
    }

    CLEANUP4:
    free(embedded_problem.elements);
    free(new_solutions);
    sapi_freeIsingResult(samples);
    CLEANUP3:
    free(embedding.elements);
    free(random_problem.elements);
    sapi_freeEmbedProblemResult(pre_embedded_problem);
    CLEANUP2:
    sapi_freeProblem(hardware_adj);
    CLEANUP1:
    sapi_freeSolver(solver);
    sapi_freeConnection(connection);
    CLEANUP0:
    sapi_globalCleanup();
    return retval;
}
