% test_bloomFilter.m

% Adicionar as pastas ao path para o Matlab encontrar o src/bloomFilter.m
addpath(genpath(pwd));

fprintf('=========================================================\n');
fprintf('     MPEI - TESTE DO MÓDULO BLOOM FILTER      \n');
fprintf('=========================================================\n\n');


% === CARREGAR DATASET DE NOTÍCIAS ===
script_dir = fileparts(mfilename('fullpath')); % Localiza a pasta 'tests'
raiz_projeto = fileparts(script_dir);          % Sobe para a raiz do projeto
addpath(genpath(raiz_projeto));                % Adiciona a 'src', 'tests' e a pasta do dataset ao MATLAB

% 2. AGORA PODE CHAMAR O FICHEIRO APENAS PELO NOME:
tabela_news = readtable('news.csv', 'VariableNamingRule', 'preserve');

% Limitar o número de registos para o teste correr eficientemente
Nu = min(2000, height(tabela_news));   

fprintf('A processar dados para %d elementos... \n', Nu);
if any(strcmp(tabela_news.Properties.VariableNames, 'Title'))
    documentos = tabela_news.Title(1:Nu);
else
    documentos = tabela_news{1:Nu, 2}; 
end

% Definir os conjuntos U1 (inserir) e U2 (testar falsos positivos)
N_inserir = floor(Nu / 2);
U1 = documentos(1:N_inserir);

% Criar elementos para U2 que garantidamente não foram inseridos
U2 = cell(N_inserir, 1);
for i = 1:N_inserir
    U2{i} = [char(U1{i}), '_not_present_in_filter'];
end

% 2. Inicialização do Filtro Bloom
fp_rate_teorico = 0.01; % Limite de 1%
bf = bloomFilter(N_inserir, fp_rate_teorico, 'classic');

% 3. Inserção de Elementos
fprintf('A inserir %d elementos no Filtro de Bloom... \n', N_inserir);
tic;
for i = 1:N_inserir
    bf = bf.insert(U1{i});
end
tempo_insercao = toc;
fprintf('Tempo gasto na inserção: %.4f segundos.\n\n', tempo_insercao);

% 4. Teste de Falsos Negativos
fprintf('A verificar integridade (Teste de Falsos Negativos em U1)... \n');
falsos_negativos = 0;
for i = 1:N_inserir
    if ~bf.lookup(U1{i})
        falsos_negativos = falsos_negativos + 1;
    end
end

% 5. Teste de Falsos Positivos
fprintf('A avaliar precisão (Teste de Falsos Positivos em U2)... \n');
falsos_positivos = 0;
tic;
for i = 1:N_inserir
    if bf.lookup(U2{i})
        falsos_positivos = falsos_positivos + 1;
    end
end
tempo_teste = toc;
fprintf('Tempo gasto na verificação de U2: %.4f segundos.\n\n', tempo_teste);

% 6. Avaliação Estatística dos Resultados
taxa_fp_real = falsos_positivos / N_inserir;

fprintf('=================== RESULTADOS DO TESTE ===================\n');
fprintf('Falsos Negativos detetados (Esperado: 0):   %d\n', falsos_negativos);
fprintf('Falsos Positivos detetados em U2:           %d\n', falsos_positivos);
fprintf('Taxa Empírica (Real) de Falsos Positivos:    %.4f\n', taxa_fp_real);
fprintf('Taxa Teórica Configurada (Limite):          %.4f\n', fp_rate_teorico);
fprintf('Tempo Total (Inserção + Teste):             %.4f segundos\n', tempo_insercao + tempo_teste);
fprintf('===========================================================\n');

% Validações de consistência por assert
assert(falsos_negativos == 0, 'Erro: Detetados falsos negativos (impossível no Bloom Filter).');
assert(taxa_fp_real <= fp_rate_teorico + 0.05, 'Erro: A taxa de falsos positivos excedeu o limite tolerável.');

fprintf('\n[MÓDULO BLOOM FILTER]: Todos os testes de volume e consistência passaram com distinção!\n');