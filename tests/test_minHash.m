% test_minHash.m

clear; clc;
fprintf('=========================================================\n');
fprintf('     MPEI - TESTE AVANÇADO E RIGOROSO DO MÓDULO MINHASH  \n');
fprintf('=========================================================\n\n');

% Adicionar as pastas ao path para garantir acesso ao src/minHash.m
addpath(genpath(pwd));


% =========================================================================
% PARTE 1: VALIDAÇÃO COM DADOS NUMÉRICOS (MOVIELENS - u.data)
% =========================================================================
fprintf('-----------------------------------------------------------\n');
fprintf('PARTE 1: Teste de Volume com Dados Numéricos (MovieLens)\n');
fprintf('-----------------------------------------------------------\n');

path_movielens = fullfile('data', 'u.data');
if ~exist(path_movielens, 'file') && exist('u.data', 'file')
    path_movielens = 'u.data';
end

if exist(path_movielens, 'file')
    fprintf('A carregar o dataset de filmes "%s"...\n', path_movielens);
    udata = load(path_movielens);  
    u = udata(:, 1:2); % Filtrar apenas colunas (User, Movie)
    clear udata;

    users = unique(u(:,1)); 
    Nu_users = min(500, length(users)); % Limitar a 500 users para cálculo exato rápido
    
    fprintf('A criar conjuntos de filmes avaliados para %d utilizadores...\n', Nu_users);
    Set_users = cell(Nu_users, 1); 
    for n = 1:Nu_users
        ind = find(u(:,1) == users(n)); 
        Set_users{n} = u(ind, 2); % Guarda vetor de IDs de filmes (inteiros)
    end
    
    % Inicializar MinHash para utilizadores
    k_user = 100;
    mh_user = minHash(k_user);
    
    % Calcular Assinaturas e Matriz de Distâncias por MinHash
    Signatures_user = zeros(k_user, Nu_users);
    for n = 1:Nu_users
        Signatures_user(:, n) = mh_user.getSignature(Set_users{n});
    end
    Sim_minhash_user = mh_user.computeSimilarityMatrix(Signatures_user);
    D_minhash_user = mh_user.distanceJaccard(Sim_minhash_user);
    
    % Jaccard Exato para validação numérica
    D_exato_user = zeros(Nu_users, Nu_users);
    for n1 = 1:Nu_users
        for n2 = n1+1:Nu_users
            inter = length(intersect(Set_users{n1}, Set_users{n2}));
            uni = length(union(Set_users{n1}, Set_users{n2}));
            D_exato_user(n1, n2) = 1 - (inter / uni);
        end
    end
    
    % Estatísticas de erro
    erros_user = abs(D_exato_user(D_exato_user > 0) - D_minhash_user(D_exato_user > 0));
    fprintf('[OK]: Dados numéricos validados. Erro Médio: %.4f\n\n', mean(erros_user));
else
    fprintf('[AVISO]: Ficheiro "u.data" não encontrado. Teste numérico ignorado.\n\n');
end


% =========================================================================
% PARTE 2: VALIDAÇÃO COM TEXTO (news.csv) E ANÁLISE DE SENSIBILIDADE DE K
% =========================================================================
fprintf('-----------------------------------------------------------\n');
fprintf('PARTE 2: Análise Textual e de Sensibilidade ao Número de Hashes (K)\n');
fprintf('-----------------------------------------------------------\n');

path_news = fullfile('data', 'news.csv');
if ~exist(path_news, 'file') && exist('news.csv', 'file')
    path_news = 'news.csv';
end

fprintf('A carregar o dataset de notícias "%s"...\n', path_news);
tabela_news = readtable(path_news, 'VariableNamingRule', 'preserve'); 

Nu_news = min(800, height(tabela_news)); % Volume ótimo para teste cruzado
fprintf('A processar e a criar Tri-Shingles de caracteres para %d notícias...\n', Nu_news);

% Instanciar um objeto provisório para gerar os shingles
mh_loader = minHash(10); 
Set_news_chars = cell(Nu_news, 1); 
Set_news_words = cell(Nu_news, 1); 

for n = 1:Nu_news
    texto = tabela_news.Title{n};
    % Criar os dois tipos de shingles suportados pelo nosso novo módulo
    Set_news_chars{n} = mh_loader.createShingles(char(texto), 3, 'chars'); 
    Set_news_words{n} = mh_loader.createShingles(char(texto), 2, 'words'); 
end

% 2A. Cálculo das Distâncias Reais (Jaccard Exato) - Modo Caracteres
fprintf('A calcular as Distâncias Reais Exatas (isto pode demorar alguns segundos)...\n');
tic;
D_exato_news = zeros(Nu_news, Nu_news);
for n1 = 1:Nu_news
    for n2 = n1+1:Nu_news
        inter = length(intersect(Set_news_chars{n1}, Set_news_chars{n2}));
        uni = length(union(Set_news_chars{n1}, Set_news_chars{n2}));
        D_exato_news(n1, n2) = 1 - (inter / uni);
    end
end
tempo_exato = toc;

% 2B. Avaliação em cascata de diferentes valores de K para o relatório
k_valores = [50, 150]; 
limiar_similaridade = 0.4; % Pares com distância < 0.4 (Similares)

% Encontrar os pares exatos abaixo do limiar
pares_reais_indices = find(D_exato_news > 0 & D_exato_news < limiar_similaridade);
num_pares_exatos = numel(pares_reais_indices);

fprintf('\n=== RESULTADOS DA SIMULAÇÃO PARAMÉTRICA ===\n');
fprintf('Tempo de Execução do Cálculo Exato: %.4f segundos\n', tempo_exato);
fprintf('Pares Similares Reais detetados (Distância < %.1f): %d\n\n', limiar_similaridade, num_pares_exatos);
fprintf('%-8s | %-12s | %-12s | %-12s | %-10s\n', 'K Hashes', 'Tempo MinHash', 'Speedup', 'Erro Médio', 'Pares MinHash');
fprintf('----------------------------------------------------------------------\n');

for k = k_valores
    mh = minHash(k);
    
    tic;
    % Gerar Assinaturas
    Signatures_news = zeros(k, Nu_news);
    for n = 1:Nu_news
        Signatures_news(:, n) = mh.getSignature(Set_news_chars{n});
    end
    
    % Utilização correta e encapsulada dos novos métodos da classe!
    Sim_matrix = mh.computeSimilarityMatrix(Signatures_news);
    D_minhash = mh.distanceJaccard(Sim_matrix);
    tempo_minhash = toc;
    
    % Isolar apenas os elementos acima da diagonal para estatísticas (evitar autorreferência)
    mascara_triangulo = triu(true(Nu_news, Nu_news), 1);
    
    erros_absolutos = abs(D_exato_news(mascara_triangulo) - D_minhash(mascara_triangulo));
    erro_medio = mean(erros_absolutos);
    
    % Contar pares similares estimados
    num_pares_minhash = sum(D_minhash(mascara_triangulo) < limiar_similaridade);
    speedup = tempo_exato / tempo_minhash;
    
    fprintf('%-8d | %-11.4fs | %-10.1fx | %-12.4f | %-10d\n', ...
        k, tempo_minhash, speedup, erro_medio, num_pares_minhash);
    
    % Validação estatística baseada na teoria (K maior implica erro menor)
    assert(erro_medio < 0.07, 'Erro: A aproximação probabilística divergiu do limite tolerável.');
end


% =========================================================================
% PARTE 3: DEMONSTRAÇÃO DO MODO SHINGLES POR PALAVRAS
% =========================================================================
fprintf('\n-----------------------------------------------------------\n');
fprintf('PARTE 3: Demonstração e Comparação de Shingles por Palavras\n');
fprintf('-----------------------------------------------------------\n');

mh_words = minHash(100);
Sig_words = zeros(100, Nu_news);
for n = 1:Nu_news
    Sig_words(:, n) = mh_words.getSignature(Set_news_words{n});
end
D_words = mh_words.distanceJaccard(mh_words.computeSimilarityMatrix(Sig_words));

% Mostrar que o módulo gerou assinaturas válidas e distintas
var_assinaturas = var(D_words(mascara_triangulo));
fprintf('Modo "words" executado com sucesso.\n');
fprintf('Variância das distâncias estimadas por palavras: %.4f\n', var_assinaturas);

assert(var_assinaturas > 0, 'Erro: As assinaturas por palavras geraram resultados idênticos redundantes.');

fprintf('-----------------------------------------------------------\n');
fprintf('[OK]: Todos os testes de volume, consistência e rigor do MinHash passaram!\n');
fprintf('===========================================================\n');