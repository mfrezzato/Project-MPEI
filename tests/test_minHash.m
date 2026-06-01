% test_minHash.m
%NMEC: 125793
%NMEC: 125487

clear; clc;
fprintf('=========================================================\n');
fprintf('     MPEI - TESTE DO MÓDULO MINHASH  \n');
fprintf('=========================================================\n\n');

% Adicionar as pastas ao path para garantir acesso ao src/minHash.m
addpath(genpath(pwd));

% carregar dados
path_news = fullfile('data', 'news.csv');
if ~exist(path_news, 'file') && exist('news.csv', 'file')
    path_news = 'news.csv';
end

fprintf('A carregar o dataset de notícias "%s"...\n', path_news);
tabela_news = readtable(path_news, 'VariableNamingRule', 'preserve'); 

% definir volume de dados
Nu_news = min(800, height(tabela_news)); 
fprintf('Dataset carregado com sucesso. Selecionados %d títulos para os testes.\n\n', Nu_news);

% validação textual
fprintf('-----------------------------------------------------------\n');
fprintf('Teste de Volume e Validação Textual Base\n');
fprintf('-----------------------------------------------------------\n');
fprintf('A processar e a criar Tri-Shingles de caracteres para as notícias...\n');

mh_loader = minHash(10); 
Set_news_chars = cell(Nu_news, 1); 
Set_news_words = cell(Nu_news, 1); 

for n = 1:Nu_news
    texto = tabela_news.Title{n};
    % criar os dois tipos de shingles suportados pelo módulo
    Set_news_chars{n} = mh_loader.createShingles(char(texto), 3, 'chars'); 
    Set_news_words{n} = mh_loader.createShingles(char(texto), 2, 'words'); 
end

% calcular jaccard teorico
fprintf('A calcular a matriz de Distâncias Reais Exatas por definição...\n');
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

% aproximação minHash k = 100
fprintf('A calcular aproximação por MinHash (K = 100)...\n');
mh_base = minHash(100);
Signatures_base = zeros(100, Nu_news);
for n = 1:Nu_news
    Signatures_base(:, n) = mh_base.getSignature(Set_news_chars{n});
end
D_minhash_base = mh_base.distanceJaccard(mh_base.computeSimilarityMatrix(Signatures_base));

mascara_triangulo = triu(true(Nu_news, Nu_news), 1);
erros_base = abs(D_exato_news(mascara_triangulo) - D_minhash_base(mascara_triangulo));

fprintf('Matrizes textuais calculadas.\n');
fprintf('   -> Tempo do Cálculo Exato:  %.4f segundos\n', tempo_exato);
fprintf('   -> Erro Absoluto Médio (K=100): %.4f\n\n', mean(erros_base));

% grafico de distribuição de densidade dos dados
figure(2);
histogram(D_exato_news(mascara_triangulo), 30, 'FaceColor', '#D95319', 'EdgeColor', 'w');
grid on;
xlabel('Distância de Jaccard Real');
ylabel('Frequência de Pares de Notícias');
title('Topologia Textual: Distribuição de Distâncias no Dataset');
fprintf('Janela Gráfica Figura 2 (Histograma de Topologia) gerada!\n\n');


fprintf('-----------------------------------------------------------\n');
fprintf('Análise de Eficiência Computacional e Erro Comparativo\n');
fprintf('-----------------------------------------------------------\n');

k_valores = [50, 150]; 
limiar_similaridade = 0.4; % pares com distância < 0.4 são considerados "parentes"

pares_reais_indices = find(D_exato_news > 0 & D_exato_news < limiar_similaridade);
num_pares_exatos = numel(pares_reais_indices);

fprintf('=== RESULTADOS DA SIMULAÇÃO PARAMÉTRICA ===\n');
fprintf('Pares Similares Reais detetados (Distância < %.1f): %d\n\n', limiar_similaridade, num_pares_exatos);
fprintf('%-8s | %-12s | %-12s | %-12s | %-10s\n', 'K Hashes', 'Tempo MinHash', 'Speedup', 'Erro Médio', 'Pares MinHash');
fprintf('----------------------------------------------------------------------\n');

for k = k_valores
    mh = minHash(k);
    
    tic;
    Signatures_news = zeros(k, Nu_news);
    for n = 1:Nu_news
        Signatures_news(:, n) = mh.getSignature(Set_news_chars{n});
    end
    
    Sim_matrix = mh.computeSimilarityMatrix(Signatures_news);
    D_minhash = mh.distanceJaccard(Sim_matrix);
    tempo_minhash = toc;
    
    erros_absolutos = abs(D_exato_news(mascara_triangulo) - D_minhash(mascara_triangulo));
    erro_medio = mean(erros_absolutos);
    
    num_pares_minhash = sum(D_minhash(mascara_triangulo) < limiar_similaridade);
    speedup = tempo_exato / tempo_minhash;
    
    fprintf('%-8d | %-11.4fs | %-10.1fx | %-12.4f | %-10d\n', ...
        k, tempo_minhash, speedup, erro_medio, num_pares_minhash);
    
    assert(erro_medio < 0.07, 'Erro: A aproximação probabilística divergiu do limite tolerável.');
end

fprintf('\n-----------------------------------------------------------\n');
fprintf('Demonstração e Comparação de Shingles por Palavras\n');
fprintf('-----------------------------------------------------------\n');

mh_words = minHash(100);
Sig_words = zeros(100, Nu_news);
for n = 1:Nu_news
    Sig_words(:, n) = mh_words.getSignature(Set_news_words{n});
end
D_words = mh_words.distanceJaccard(mh_words.computeSimilarityMatrix(Sig_words));

var_assinaturas = var(D_words(mascara_triangulo));
fprintf('Modo "words" executado com sucesso.\n');
fprintf('Variância das distâncias estimadas por palavras: %.4f\n', var_assinaturas);

assert(var_assinaturas > 0, 'Erro: As assinaturas por palavras falharam.');


fprintf('\n-----------------------------------------------------------\n');
fprintf('Análise Contínua de K e Curva de Erro Estatístico\n');
fprintf('-----------------------------------------------------------\n');
fprintf('A correr varredura de K para traçar a curva de convergência... (Aguarde)\n');

k_passos = [10, 30, 60, 100, 150, 200, 300];
erros_k = zeros(1, numel(k_passos));

for idx_k = 1:numel(k_passos)
    k_atual = k_passos(idx_k);
    mh_curva = minHash(k_atual);
    
    Signatures_curva = zeros(k_atual, Nu_news);
    for n = 1:Nu_news
        Signatures_curva(:, n) = mh_curva.getSignature(Set_news_chars{n});
    end
    D_curva = mh_curva.distanceJaccard(mh_curva.computeSimilarityMatrix(Signatures_curva));
    
    erros_abs = abs(D_exato_news(mascara_triangulo) - D_curva(mascara_triangulo));
    erros_k(idx_k) = mean(erros_abs);
end

figure(1);
plot(k_passos, erros_k, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 6);
grid on;
xlabel('Número de Funções de Hash (k)');
ylabel('Erro Absoluto Médio de Jaccard');
title('Curva de Convergência Estatística do MinHash');
fprintf('Janela Gráfica Figura 1 (Curva de Erro) gerada!\n\n');

fprintf('-----------------------------------------------------------\n');
fprintf('Análise de Impacto do Comprimento do Shingle (PL7 Ex 7.2)\n');
fprintf('-----------------------------------------------------------\n');

shingle_tamanhos = [2, 3, 4, 5];
dist_medias_shingle = zeros(1, numel(shingle_tamanhos));

fprintf('%-15s | %-25s | %-20s\n', 'Tamanho Shingle', 'Total de Shingles Únicos', 'Distância Média Jaccard');
fprintf('-----------------------------------------------------------\n');

% iubconjunto de 200 itens para cálculo instantâneo da sensibilidade do shingle
Nu_sub = 200; 

for s = 1:numel(shingle_tamanhos)
    size_atual = shingle_tamanhos(s);
    
    Set_temp = cell(Nu_sub, 1);
    all_shingles_vocab = {};
    for n = 1:Nu_sub
        shingles_doc = mh_loader.createShingles(char(tabela_news.Title{n}), size_atual, 'chars');
        Set_temp{n} = shingles_doc;
        all_shingles_vocab = [all_shingles_vocab, shingles_doc]; %#ok<AGROW>
    end
    
    total_vocab_shingles = numel(unique(all_shingles_vocab));
    
    D_sub_exato = zeros(Nu_sub, Nu_sub);
    for n1 = 1:Nu_sub
        for n2 = n1+1:Nu_sub
            inter = length(intersect(Set_temp{n1}, Set_temp{n2}));
            uni = length(union(Set_temp{n1}, Set_temp{n2}));
            D_sub_exato(n1, n2) = 1 - (inter / uni);
        end
    end
    
    mascara_sub = triu(true(Nu_sub, Nu_sub), 1);
    dist_medias_shingle(s) = mean(D_sub_exato(mascara_sub));
    
    fprintf('%-15d | %-25d | %-20.4f\n', size_atual, total_vocab_shingles, dist_medias_shingle(s));
end

% grafico impacto shingles
figure(3);
plot(shingle_tamanhos, dist_medias_shingle, '-^', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'Color', [0 0.5 0]);
grid on;
set(gca, 'XTick', shingle_tamanhos);
xlabel('Comprimento do Shingle de Caracteres');
ylabel('Distância Média de Jaccard Real');
title('Impacto do Tamanho do Shingle na Densidade de Similaridade');
fprintf('Janela Gráfica Figura 3 (Análise de Shingle) gerada!\n');

fprintf('-----------------------------------------------------------\n');
fprintf('Todos os testes de volume, consistência e rigor do MinHash passaram!\n');
fprintf('===========================================================\n');