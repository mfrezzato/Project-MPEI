% test_bloomFilter.m

fprintf('=========================================================\n');
fprintf('     MPEI - TESTE DO MÓDULO Bloom Filter      \n');
fprintf('=========================================================\n\n');

% Adicionar as pastas ao path para o Matlab encontrar o src/bloomFilter.m
addpath(genpath(pwd));

% === CARREGAR DATASET DE NOTÍCIAS ===
tabela_news = readtable('news.csv', 'VariableNamingRule', 'preserve'); 

Nu = min(2000, height(tabela_news));   
documentos = tabela_news.Title(1:Nu);

N_inserir = floor(Nu / 2);
U1 = documentos(1:N_inserir);

% Criar U2 de forma vetorizada
U2 = strcat(U1, '_not_present');

% Inicialização e Inserção
bf = bloomFilter(N_inserir, 0.01, 'classic');
for i = 1:N_inserir
    bf = bf.insert(U1{i});
end

% Teste de Falsos Negativos e Positivos
falsos_negativos = 0;
falsos_positivos = 0;

for i = 1:N_inserir
    if ~bf.lookup(U1{i}), falsos_negativos = falsos_negativos + 1; end
    if bf.lookup(U2{i}),  falsos_positivos = falsos_positivos + 1; end
end

% Avaliação
taxa_fp_real = falsos_positivos / N_inserir;

fprintf('=== RESULTADOS DO TESTE ===\n');
fprintf('Falsos Negativos: %d (Esp: 0)\n', falsos_negativos);
fprintf('Taxa FP Real:     %.4f (Teórica: 0.01)\n', taxa_fp_real);

assert(falsos_negativos == 0, 'Erro: Detetados falsos negativos.');
assert(taxa_fp_real <= 0.06, 'Erro: Taxa de falsos positivos excedeu o limite.');
fprintf('[OK]: Todos os testes passaram!\n');