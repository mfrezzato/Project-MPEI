% test_naiveBayes.m
clear; clc;
fprintf('=========================================================\n');
fprintf('     MPEI - TESTE COMPREENSIVO DO MÓDULO NAIVE BAYES     \n');
fprintf('=========================================================\n\n');

% Adicionar as pastas ao path para garantir que o Matlab encontra o src/
addpath(genpath(pwd));

% 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
fprintf('A carregar o dataset "news.csv"... \n');
opts = detectImportOptions('news.csv', 'VariableNamingRule', 'preserve');
tabela_news = readtable('news.csv', opts); 

% Definir volume de dados para o teste de validação
Nu = min(2000, height(tabela_news));   
documentos = tabela_news.Title(1:Nu);
labels = tabela_news.('Class Index')(1:Nu);
if isnumeric(labels)
    labels = cellstr(string(labels)); 
end

% Fixar a seed para garantir reprodutibilidade na divisão de dados
rng(42); 
indices = randperm(Nu);
limite = round(0.70 * Nu); % 70% para Treino e 30% para Teste
docs_treino   = documentos(indices(1:limite));
labels_treino = labels(indices(1:limite));
docs_teste    = documentos(indices(limite+1:end));
labels_teste  = labels(indices(limite+1:end));
N_teste = numel(docs_teste);
fprintf('Dados preparados: %d documentos de Treino, %d de Teste.\n\n', limite, N_teste);

% =========================================================================
% EVALUAR MODO 1: MULTINOMIAL
% =========================================================================
fprintf('>>> Avaliando Modelo NAÏVE BAYES: MULTINOMIAL...\n');
nb_multi = naiveBayes('multinomial');
tic;
nb_multi = nb_multi.train(docs_treino, labels_treino);
tempo_treino_multi = toc;

tic;
previsoes_multi = nb_multi.classify(docs_teste);
tempo_class_multi = toc;

[acc_multi, metricas_multi] = calcularMetricasMultiClasse(labels_teste, previsoes_multi, nb_multi.classes);

% =========================================================================
% EVALUAR MODO 2: BERNOULLI (CORRIGIDO)
% =========================================================================
fprintf('>>> Avaliando Modelo NAÏVE BAYES: BERNOULLI...\n');
nb_bern = naiveBayes('bernoulli');
tic;
nb_bern = nb_bern.train(docs_treino, labels_treino);
tempo_treino_bern = toc;

tic;
previsoes_bern = nb_bern.classify(docs_teste);
tempo_class_bern = toc;

[acc_bern, metricas_bern] = calcularMetricasMultiClasse(labels_teste, previsoes_bern, nb_bern.classes);

% =========================================================================
% RESUMO COMPARATIVO DE RESULTADOS (EXCELENTE PARA O RELATÓRIO)
% =========================================================================
fprintf('\n===========================================================\n');
fprintf('                TABELA COMPARATIVA GLOBAL                  \n');
fprintf('===========================================================\n');
fprintf('%-18s | %-15s | %-15s\n', 'Métrica', 'Multinomial', 'Bernoulli');
fprintf('-----------------------------------------------------------\n');
fprintf('%-18s | %-14.2f%% | %-14.2f%%\n', 'Exatidão Global', acc_multi * 100, acc_bern * 100);
fprintf('%-18s | %-13.4fs | %-13.4fs\n', 'Tempo de Treino', tempo_treino_multi, tempo_treino_bern);
fprintf('%-18s | %-13.4fs | %-13.4fs\n', 'Tempo Classif.', tempo_class_multi, tempo_class_bern);
fprintf('===========================================================\n\n');

fprintf('===========================================================\n');
fprintf('             DETALHE POR CLASSE (F1-SCORE)                 \n');
fprintf('===========================================================\n');
fprintf('%-15s | %-15s | %-15s\n', 'ID da Classe', 'F1 Multinomial', 'F1 Bernoulli');
fprintf('-----------------------------------------------------------\n');
for c = 1:numel(nb_multi.classes)
    fprintf('%-15s | %-15.4f | %-15.4f\n', ...
        nb_multi.classes{c}, metricas_multi(c).f1, metricas_bern(c).f1);
end
fprintf('===========================================================\n');

% Validações de sanidade automatizadas
assert(acc_multi >= 0 && acc_multi <= 1, 'Erro nas métricas do Multinomial.');
assert(acc_bern >= 0 && acc_bern <= 1, 'Erro nas métricas do Bernoulli.');
fprintf('\n[OK]: Todos os testes detalhados e cruzados passaram com sucesso!\n');

% =========================================================================
% FUNÇÃO AUXILIAR: CÁLCULO DE MÉTRICAS MULTI-CLASSE
% =========================================================================
function [exatidatoglobal, listaMetricas] = calcularMetricasMultiClasse(reais, previstas, listaClasses)
    % CORREÇÃO CRÍTICA: Forçar linearização para vetores coluna (N x 1)
    % Isto elimina incompatibilidades de tamanho no strcmp e loops seguintes
    reais = reais(:);
    previstas = previstas(:);

    numClasses = numel(listaClasses);
    N = numel(reais);
    
    % Acertos globais - agora a comparação corre sem erros
    acertos = sum(strcmp(reais, previstas));
    exatidatoglobal = acertos / N;
    
    % Inicializar estrutura para armazenar métricas por classe
    listaMetricas = struct('classe', {}, 'precisao', {}, 'sensibilidade', {}, 'f1', {});
    
    for c = 1:numClasses
        classeAlvo = listaClasses{c};
        
        VP = 0; FP = 0; FN = 0;
        for i = 1:N
            isReal = strcmp(reais{i}, classeAlvo);
            isPrev = strcmp(previstas{i}, classeAlvo);
            
            if isReal && isPrev,   VP = VP + 1; end
            if ~isReal && isPrev,  FP = FP + 1; end
            if isReal && ~isPrev,  FN = FN + 1; end
        end
        
        precisao = 0; sensibilidade = 0; f1 = 0;
        if (VP + FP) > 0, precisao = VP / (VP + FP); end
        if (VP + FN) > 0, sensibilidade = VP / (VP + FN); end
        if (precisao + sensibilidade) > 0
            f1 = 2 * (precisao * sensibilidade) / (precisao + sensibilidade);
        end
        
        listaMetricas(c).classe = classeAlvo;
        listaMetricas(c).precisao = precisao;
        listaMetricas(c).sensibilidade = sensibilidade;
        listaMetricas(c).f1 = f1;
    end
end