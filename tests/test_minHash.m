% test_minHash.m

% Adicionar as pastas ao path para o Matlab encontrar o src/minHash.m
addpath(genpath(pwd));

fprintf('=========================================================\n');
fprintf('     MPEI - TESTE DO MÓDULO MINHASH      \n');
fprintf('=========================================================\n\n');

% Carregar e Preparar os Dados (MovieLens)
if ~exist('u.data', 'file')
    error('Erro: O ficheiro "u.data" não foi encontrado na pasta raiz do projeto.');
end

fprintf('A carregar o dataset "u.data"... \n');
udata = load("u.data");  
u = udata(:, 1:2);      % Fica apenas com as duas primeiras colunas (User, Movie)
clear udata;

users = unique(u(:,1)); % Extrai os IDs únicos dos utilizadores 
Nu = length(users);     % Número total de utilizadores 

fprintf('A processar os conjuntos de filmes para %d utilizadores...\n\n', Nu);
Set = cell(Nu, 1); % Constroi a lista de filmes avaliados por cada user 
for n = 1:Nu
    ind = find(u(:,1) == users(n)); 
    Set{n} = u(ind, 2);             % Guarda os IDs dos filmes num array de células 
end

% Cálculo das Distâncias Reais (Jaccard Exato)
fprintf('Calculando Distâncias Reais (Jaccard Exato)...\n');
tic;
J_exato = zeros(Nu, Nu);
for n1 = 1:Nu
    for n2 = n1+1:Nu
        % Fórmula teórica: 1 - (interseção / união) 
        inter = length(intersect(Set{n1}, Set{n2}));
        uni = length(union(Set{n1}, Set{n2}));
        J_exato(n1, n2) = 1 - (inter / uni);
    end
end
tempo_exato = toc;
fprintf('Tempo gasto no cálculo exato: %.4f segundos.\n\n', tempo_exato);

% Gerar e armazenar as assinaturas de todos os utilizadores
Signatures = zeros(k_hash, Nu);
tic;
for n = 1:Nu
    Signatures(:, n) = mh.getSignature(Set{n});
end

% Comparar os pares de assinaturas (Vetorizado - Elimina overhead de loops da classe)
J_minhash = zeros(Nu, Nu);
for n1 = 1:Nu
    if n1 < Nu
        coincidencias = sum(Signatures(:, n1) == Signatures(:, n1+1:Nu), 1);
        J_minhash(n1, n1+1:Nu) = 1 - (coincidencias / k_hash);
    end
end
tempo_minhash = toc;
fprintf('Tempo gasto com MinHash: %.4f segundos.\n\n', tempo_minhash);

% Avaliação Estatística dos Resultados 
limiar = 0.4;

pares_exatos = 0;
pares_minhash = 0;
erros_absolutos = [];

for n1 = 1:Nu
    for n2 = n1+1:Nu
        dist_e = J_exato(n1, n2);
        dist_m = J_minhash(n1, n2);
        
        erros_absolutos(end+1) = abs(dist_e - dist_m);
        
        if dist_e < limiar
            pares_exatos = pares_exatos + 1;
        end
        if dist_m < limiar
            pares_minhash = pares_minhash + 1;
        end
    end
end

erro_medio = mean(erros_absolutos);
max_erro = max(erros_absolutos);

fprintf('=================== RESULTADOS DO TESTE ===================\n');
fprintf('Pares similares detetados pelo Jaccard Real:  %d\n', pares_exatos);
fprintf('Pares similares detetados pelo teu MinHash:   %d\n', pares_minhash);
fprintf('Erro Absoluto Médio da aproximação:           %.4f\n', erro_medio);
fprintf('Erro Máximo registado no teste:               %.4f\n', max_erro);
fprintf('Ganho de Performance com MinHash:             %.1fx mais rápido\n', (tempo_exato / tempo_minhash));
fprintf('===========================================================\n');

% Validação automática para garantir que o teste passou
assert(erro_medio < 0.05, 'O erro médio do estimador probabilístico está muito acima do tolerável.');
fprintf('\n[MÓDULO MINHASH]: Todos os testes de volume e consistência passaram com distinção!\n');