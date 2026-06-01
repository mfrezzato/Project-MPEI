% test_naiveBayes.m
%NMEC: 125793
%NMEC: 125487

clear; clc;
fprintf('=========================================================\n');
fprintf('     MPEI - TESTE DO MÓDULO NAIVE BAYES     \n');
fprintf('=========================================================\n\n');

% Adicionar as pastas ao path para garantir que o Matlab encontra o src/
addpath(genpath(pwd));

% carregar dados
fprintf('A carregar o dataset "news.csv"... \n');
opts = detectImportOptions('news.csv', 'VariableNamingRule', 'preserve');
tabela_news = readtable('news.csv', opts); 

% definir volumes para validação
Nu = min(2000, height(tabela_news));   
documentos = tabela_news.Title(1:Nu);
labels = tabela_news.('Class Index')(1:Nu);
if isnumeric(labels)
    labels = cellstr(string(labels)); 
end

% fixar a seed para garantir reprodutibilidade na divisão de dados
rng(42); 
indices = randperm(Nu);
limite = round(0.70 * Nu); % 70% para Treino e 30% para Teste
docs_treino   = documentos(indices(1:limite));
labels_treino = labels(indices(1:limite));
docs_teste    = documentos(indices(limite+1:end));
labels_teste  = labels(indices(limite+1:end));
N_teste = numel(docs_teste);
fprintf('Dados preparados: %d documentos de Treino, %d de Teste.\n\n', limite, N_teste);

% avaliar modo Mutinomial do modulo
fprintf('>>> Avaliando Modelo NAÏVE BAYES: MULTINOMIAL...\n');
nb_multi = naiveBayes('multinomial');
tic;
nb_multi = nb_multi.train(docs_treino, labels_treino);
tempo_treino_multi = toc;

tic;
previsoes_multi = nb_multi.classify(docs_teste);
tempo_class_multi = toc;

[acc_multi, metricas_multi] = calcularMetricasMultiClasse(labels_teste, previsoes_multi, nb_multi.classes);

% avaliar modo bernoulli do modulo
fprintf('>>> Avaliando Modelo NAÏVE BAYES: BERNOULLI...\n');
nb_bern = naiveBayes('bernoulli');
tic;
nb_bern = nb_bern.train(docs_treino, labels_treino);
tempo_treino_bern = toc;

tic;
previsoes_bern = nb_bern.classify(docs_teste);
tempo_class_bern = toc;

[acc_bern, metricas_bern] = calcularMetricasMultiClasse(labels_teste, previsoes_bern, nb_bern.classes);

% tabelas de comparação
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
fprintf('             DETALHE POR CLASSE               \n');
fprintf('===========================================================\n');
fprintf('%-15s | %-15s | %-15s\n', 'ID da Classe', 'F1 Multinomial', 'F1 Bernoulli');
fprintf('-----------------------------------------------------------\n');
for c = 1:numel(nb_multi.classes)
    fprintf('%-15s | %-15.4f | %-15.4f\n', ...
        nb_multi.classes{c}, metricas_multi(c).f1, metricas_bern(c).f1);
end
fprintf('===========================================================\n');

% asserts para caso haja erros
assert(acc_multi >= 0 && acc_multi <= 1, 'Erro nas métricas do Multinomial.');
assert(acc_bern >= 0 && acc_bern <= 1, 'Erro nas métricas do Bernoulli.');
fprintf('\nBlocos base validados com sucesso!\n\n');


fprintf('-----------------------------------------------------------\n');
fprintf('Geração Visual da Matriz de Confusão (Multinomial)\n');
fprintf('-----------------------------------------------------------\n');

% labels das categorias
classNames_labels = {'Mundo', 'Desporto', 'Economia', 'Sci/Tech'};

% mapeamento das categorias a IDs
labels_reais_cat = categorical(cellstr(labels_teste), {'1','2','3','4'}, classNames_labels);
previsoes_multi_cat = categorical(cellstr(previsoes_multi), {'1','2','3','4'}, classNames_labels);

figure(2);
confusionchart(labels_reais_cat, previsoes_multi_cat, ...
    'Title', 'Matriz de Confusão - Naïve Bayes Multinomial');
fprintf('Janela Gráfica Figura 2 gerada com sucesso!\n\n');


fprintf('-----------------------------------------------------------\n');
fprintf('Análise de Sensibilidade ao Volume de Treino\n');
fprintf('-----------------------------------------------------------\n');
fprintf('A avaliar a evolução da exatidão com o crescimento do treino...\n');

volumes_treino = [100, 300, 600, 1000, limite];
acc_evolucao = zeros(1, numel(volumes_treino));
 
for v = 1:numel(volumes_treino)
    tamanho_atual = volumes_treino(v);
     
    % instanciar e treinar um classificador temporário com amostras reduzidas
    nb_temp = naiveBayes('multinomial');
    nb_temp = nb_temp.train(docs_treino(1:tamanho_atual), labels_treino(1:tamanho_atual));
     
    % classificar o mesmo conjunto de teste fixo para manter a consistência estatística
    prev_temp = nb_temp.classify(docs_teste);
    [acc_evolucao(v), ~] = calcularMetricasMultiClasse(labels_teste, prev_temp, nb_temp.classes);
end
 
% grafico curva de aprendizagem
figure(3);
plot(volumes_treino, acc_evolucao * 100, '-^', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'Color', [0 0.5 0]);
grid on;
xlabel('Número de Documentos Utilizados no Treino');
ylabel('Exatidão Global no Teste (%)');
title('Curva de Aprendizagem do Classificador (Naïve Bayes)');
fprintf('Janela Gráfica Figura 3 gerada com sucesso!\n\n');


fprintf('-----------------------------------------------------------\n');
fprintf('Inspeção de Conhecimento - Top 5 Palavras por Categoria\n');
fprintf('-----------------------------------------------------------\n');

for c = 1:numel(nb_multi.classes)
    id_classe_original = nb_multi.classes{c};
    nome_amigavel = classMap_Nome(id_classe_original);
    
    pesos_classe = nb_multi.logLikelihood(c, :);
    
    % ordenar de forma decrescente para obter os termos mais relevantes
    [~, indices_ordenados] = sort(pesos_classe, 'descend');
    
    fprintf('Top 5 Palavras Relevantes para [Classe %s - %s]:\n', id_classe_original, nome_amigavel);
    for w = 1:5
        idx_palavra = indices_ordenados(w);
        palavra = nb_multi.vocabulary{idx_palavra};
        fprintf('   %d. "%s" (Log-Likelihood: %.4f)\n', w, palavra, pesos_classe(idx_palavra));
    end
    fprintf('\n');
end
fprintf('-----------------------------------------------------------\n');
fprintf('Todos os testes avançados e criativos do Naïve Bayes terminaram!\n');
fprintf('===========================================================\n');


% funções auxiliares

function nome = classMap_Nome(id)
    switch id
        case '1', nome = 'Mundo';
        case '2', nome = 'Desporto';
        case '3', nome = 'Economia';
        case '4', nome = 'Ciência/Tecnologia';
        otherwise, nome = 'Desconhecida';
    end
end

function [exatidatoglobal, listaMetricas] = calcularMetricasMultiClasse(reais, previstas, listaClasses)
    reais = reais(:);
    previstas = previstas(:);
    numClasses = numel(listaClasses);
    N = numel(reais);
    
    acertos = sum(strcmp(reais, previstas));
    exatidatoglobal = acertos / N;
    
    % estrutura para armazenar dados por classe
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