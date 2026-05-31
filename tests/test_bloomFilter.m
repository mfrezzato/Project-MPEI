% test_bloomFilter.m

clear; clc;
fprintf('=========================================================\n');
fprintf('     MPEI - TESTE COMPREENSIVO DO MÓDULO BLOOM FILTER    \n');
fprintf('=========================================================\n\n');

% Adicionar as pastas ao path para o Matlab encontrar o src/bloomFilter.m
addpath(genpath(pwd));

% 1. CARREGAMENTO E DIVISÃO DOS DADOS REAIS
fprintf('A carregar o dataset "news.csv"... \n');
tabela_news = readtable('news.csv', 'VariableNamingRule', 'preserve'); 

Nu = min(3000, height(tabela_news));   
documentos = tabela_news.Title(1:Nu);

% Garantir independência estatística:
% U1: Elementos a INSERIR (Primeira metade)
% U2: Elementos NÃO INSERIDOS para testar Falsos Positivos (Segunda metade)
N_inserir = floor(Nu / 2);
U1 = documentos(1:N_inserir);
U2 = documentos(N_inserir+1:2*N_inserir);

fprintf('Dados preparados: %d itens para inserção e %d itens independentes para teste de FP.\n\n', N_inserir, N_inserir);


% =========================================================================
% PARTE 1: VALIDAÇÃO DO BLOOM FILTER CLÁSSICO E ANÁLISE TEÓRICA
% =========================================================================
fprintf('-----------------------------------------------------------\n');
fprintf('PARTE 1: Teste do Filtro Clássico sob Diferentes Limites Teóricos\n');
fprintf('-----------------------------------------------------------\n');

% Vetor com diferentes taxas de falsos positivos (fpRate) para testar o comportamento
fp_teoricos = [0.10, 0.01, 0.001];

fprintf('%-12s | %-12s | %-10s | %-12s | %-12s\n', 'FP Teórico', 'Num Bits(m)', 'Hashes(k)', 'Falsos Neg.', 'FP Real');
fprintf('-----------------------------------------------------------\n');

for f = 1:numel(fp_teoricos)
    fp_alvo = fp_teoricos(f);
    
    % Inicialização com o redimensionamento ótimo corrigido
    bf_classic = bloomFilter(N_inserir, fp_alvo, 'classic');
    
    % Inserção
    for i = 1:N_inserir
        bf_classic = bf_classic.insert(U1{i});
    end
    
    % Teste de Falsos Negativos (U1 deve dar sempre True)
    falsos_negativos = 0;
    for i = 1:N_inserir
        if ~bf_classic.lookup(U1{i})
            falsos_negativos = falsos_negativos + 1;
        end
    end
    
    % Teste de Falsos Positivos (U2 nunca foi inserido, logo True = Falso Positivo)
    falsos_positivos = 0;
    for i = 1:N_inserir
        if bf_classic.lookup(U2{i})
            falsos_positivos = falsos_positivos + 1;
        end
    end
    
    taxa_fp_real = falsos_positivos / N_inserir;
    
    fprintf('%-12.3f | %-12d | %-10d | %-12d | %-12.4f\n', ...
        fp_alvo, bf_classic.numBits, bf_classic.numHashes, falsos_negativos, taxa_fp_real);
    
    % Validações matemáticas estritas
    assert(falsos_negativos == 0, 'Erro Crítico: O Bloom Filter gerou falsos negativos!');
    assert(taxa_fp_real <= fp_alvo + 0.02, 'Erro: A taxa de falsos positivos real diverge muito da teórica.');
end
fprintf('[OK]: Filtro Clássico validado matematicamente com sucesso.\n\n');


% =========================================================================
% PARTE 2: VALIDAÇÃO DO COUNTING BLOOM FILTER E REMOÇÃO
% =========================================================================
fprintf('-----------------------------------------------------------\n');
fprintf('PARTE 2: Teste de Integridade do Counting Bloom Filter\n');
fprintf('-----------------------------------------------------------\n');

fp_teste_counting = 0.01;
bf_counting = bloomFilter(N_inserir, fp_teste_counting, 'counting');

fprintf('A inserir elementos no Counting Bloom Filter...\n');
for i = 1:N_inserir
    bf_counting = bf_counting.insert(U1{i});
end

% Escolher uma amostra para testar a remoção (ex: os primeiros 100 elementos)
N_remoção = min(100, N_inserir);
fprintf('A testar a remoção de %d elementos...\n', N_remoção);

% Verificar que eles existem antes de remover
todos_existiam = true;
for i = 1:N_remoção
    if ~bf_counting.lookup(U1{i})
        todos_existiam = false;
    end
end

% Aplicar a remoção
for i = 1:N_remoção
    bf_counting = bf_counting.remove(U1{i});
end

% Verificar se foram removidos com sucesso (lookup deve falhar para a maioria,
% salvaguardando raras colisões inevitáveis de falsos positivos)
removidos_com_sucesso = 0;
for i = 1:N_remoção
    if ~bf_counting.lookup(U1{i})
        removidos_com_sucesso = removidos_com_sucesso + 1;
    end
end

taxa_sucesso_remocao = removidos_com_sucesso / N_remoção;

fprintf('\n=== RESULTADOS MÓDULO COUNTING ===\n');
fprintf('Presença confirmada pré-remoção: %s\n', char(string(todos_existiam)));
fprintf('Elementos Removidos Testados:    %d\n', N_remoção);
fprintf('Elementos que desapareceram:     %d (Taxa: %.2f%%)\n', removidos_com_sucesso, taxa_sucesso_remocao * 100);

assert(todos_existiam, 'Erro: Elementos inseridos não foram encontrados no Counting Bloom Filter.');
assert(taxa_sucesso_remocao >= 0.95, 'Erro: A remoção falhou ou houve um nível anormal de colisões.');

fprintf('-----------------------------------------------------------\n');
fprintf('[OK]: Todos os testes do módulo Bloom Filter passaram!\n');
fprintf('===========================================================\n');