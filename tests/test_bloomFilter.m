% test_bloomFilter.m
%NMEC: 125793
%NMEC: 125487

clear; clc;
fprintf('=========================================================\n');
fprintf('     MPEI - TESTE MÓDULO BLOOM FILTER    \n');
fprintf('=========================================================\n\n');

addpath(genpath(pwd));

% carregar e dividir os dados
fprintf('A carregar o dataset "news.csv"... \n');
tabela_news = readtable('news.csv', 'VariableNamingRule', 'preserve'); 
Nu = min(3000, height(tabela_news));   
documentos = cellstr(tabela_news.Title(1:Nu));

N_inserir = floor(Nu / 2);
U1 = documentos(1:N_inserir);
U2 = documentos(N_inserir+1:2*N_inserir);
fprintf('Dados preparados: %d itens para inserção e %d de teste.\n\n', N_inserir, N_inserir);

% validação do bloom filter classic
fprintf('-----------------------------------------------------------\n');
fprintf('Teste do Filtro Clássico sob Diferentes Limites Teóricos\n');
fprintf('-----------------------------------------------------------\n');
fp_teoricos = [0.10, 0.01, 0.001];
fprintf('%-12s | %-12s | %-10s | %-12s | %-12s\n', 'FP Teórico', 'Num Bits(m)', 'Hashes(k)', 'Falsos Neg.', 'FP Real');
fprintf('-----------------------------------------------------------\n');

for f = 1:numel(fp_teoricos)
    fp_alvo = fp_teoricos(f);
    bf_classic = bloomFilter(N_inserir, fp_alvo, 'classic');
    
    for i = 1:N_inserir
        bf_classic = bf_classic.insert(U1{i});
    end
    
    falsos_negativos = 0;
    for i = 1:N_inserir
        if ~bf_classic.lookup(U1{i}), falsos_negativos = falsos_negativos + 1; end
    end
    
    falsos_positivos = 0;
    for i = 1:N_inserir
        if bf_classic.lookup(U2{i}), falsos_positivos = falsos_positivos + 1; end
    end
    
    taxa_fp_real = falsos_positivos / N_inserir;
    fprintf('%-12.3f | %-12d | %-10d | %-12d | %-12.4f\n', ...
        fp_alvo, bf_classic.numBits, bf_classic.numHashes, falsos_negativos, taxa_fp_real);
    
    assert(falsos_negativos == 0, 'Erro Crítico: O Bloom Filter gerou falsos negativos!');
    assert(taxa_fp_real <= fp_alvo + 0.03, 'Erro: A taxa de falsos positivos excedeu a tolerância.');
end
fprintf('Filtro Clássico validado com sucesso.\n\n');

% validação do bloom filter counting e remoção
fprintf('-----------------------------------------------------------\n');
fprintf('Teste de Integridade do Counting Bloom Filter\n');
fprintf('-----------------------------------------------------------\n');
bf_counting = bloomFilter(N_inserir, 0.01, 'counting');
for i = 1:N_inserir, bf_counting = bf_counting.insert(U1{i}); end

N_remocao = min(100, N_inserir);
todos_existiam = true;
for i = 1:N_remocao
    if ~bf_counting.lookup(U1{i}), todos_existiam = false; end
end

for i = 1:N_remocao, bf_counting = bf_counting.remove(U1{i}); end

removidos_com_sucesso = 0;
for i = 1:N_remocao
    if ~bf_counting.lookup(U1{i}), removidos_com_sucesso = removidos_com_sucesso + 1; end
end
taxa_sucesso_remocao = removidos_com_sucesso / N_remocao;

fprintf('Presença pré-remoção: %s | Itens testados: %d | Removidos: %d (Taxa: %.2f%%)\n', ...
    char(string(todos_existiam)), N_remocao, removidos_com_sucesso, taxa_sucesso_remocao * 100);
assert(todos_existiam && taxa_sucesso_remocao >= 0.95, 'Erro no Counting Bloom Filter.');
fprintf('Módulo Counting validado com sucesso.\n\n');

% analise empirica do K otimo
fprintf('-----------------------------------------------------------\n');
fprintf('Teste de Sensibilidade - Determinação do K Ótimo\n');
fprintf('-----------------------------------------------------------\n');
fprintf('A simular variação de K para um tamanho fixo de filtro... (Aguarde)\n');

m_fixo = 10000; % Fixamos o tamanho em bits do filtro
k_testados = 1:12;
fp_registados = zeros(1, numel(k_testados));

for idx_k = 1:numel(k_testados)
    k_atual = k_testados(idx_k);
    
    % instanciação manual alterando a propriedade diretamente para teste de stress
    bf_custom = bloomFilter(N_inserir, 0.05, 'classic');
    bf_custom.numBits = m_fixo;
    bf_custom.numHashes = k_atual;
    bf_custom.bits = false(1, m_fixo); % reinicializa o vetor com m_fixo
    
    for i = 1:N_inserir, bf_custom = bf_custom.insert(U1{i}); end
    
    falsos_pos = 0;
    for i = 1:N_inserir
        if bf_custom.lookup(U2{i}), falsos_pos = falsos_pos + 1; end
    end
    fp_registados(idx_k) = falsos_pos / N_inserir;
end

% determinar o k ótimo experimental
[min_fp, idx_otimo] = min(fp_registados);
k_otimo_teorico = round((m_fixo / N_inserir) * log(2));

fprintf('Resultados de Sensibilidade:\n');
fprintf('-> K Ótimo Experimental: %d (Menor taxa de FP: %.4f)\n', k_testados(idx_otimo), min_fp);
fprintf('-> K Ótimo Teórico:      %d\n', k_otimo_teorico);

% gerar o Gráfico para o Relatório Técnico
figure(1);
plot(k_testados, fp_registados * 100, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
grid on;
xlabel('Número de Funções Hash (k)');
ylabel('Taxa de Falsos Positivos (%)');
title('Análise Empírica do K Ótimo (Filtro de Bloom)');
fprintf('Janela Gráfica Figura 1 gerada com sucesso!\n');
fprintf('-----------------------------------------------------------\n');
fprintf('Todos os testes avançados do Bloom Filter passaram!\n');