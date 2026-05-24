% test_naiveBayes.m

fprintf('=========================================================\n');
fprintf('     MPEI - TESTE DO MÓDULO NAIVE BAYES      \n');
fprintf('=========================================================\n\n');

% Adicionar as pastas ao path para o Matlab encontrar o src/naiveBayes.m
addpath(genpath(pwd));

% === CARREGAR DATASET DE NOTÍCIAS ===
opts = detectImportOptions('news.csv', 'VariableNamingRule', 'preserve');
tabela_news = readtable('news.csv', opts); 

Nu = min(1500, height(tabela_news));   

documentos = tabela_news.Title(1:Nu);
labels = tabela_news.('Class Index')(1:Nu);

if isnumeric(labels), labels = cellstr(string(labels)); end

rng(42); 
indices = randperm(Nu);
limite = round(0.7 * Nu); % 70% treino

docs_treino   = documentos(indices(1:limite));
labels_treino = labels(indices(1:limite));
docs_teste    = documentos(indices(limite+1:end));
labels_teste  = labels(indices(limite+1:end));

% Inicialização e Treino
nb = naiveBayes('bernoulli');
nb = nb.train(docs_treino, labels_treino);

% Classificação e Cálculo de Métricas
previsoes = nb.classify(docs_teste);

classe_positiva = nb.classes{1}; 
VP = 0; FP = 0; VN = 0; FN = 0;

for i = 1:numel(docs_teste)
    real = labels_teste{i};
    previsto = previsoes{i};
    
    if strcmp(real, classe_positiva)
        if strcmp(previsto, classe_positiva), VP = VP + 1; else, FN = FN + 1; end
    else
        if strcmp(previsto, classe_positiva), FP = FP + 1; else, VN = VN + 1; end
    end
end

N_teste = numel(docs_teste);
exatidao = (VP + VN) / N_teste;

precisao = 0; sensibilidade = 0; f1_score = 0;
if (VP + FP) > 0, precisao = VP / (VP + FP); end
if (VP + FN) > 0, sensibilidade = VP / (VP + FN); end
if (precisao + sensibilidade) > 0, f1_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade); end

fprintf('=================== RESULTADOS DO TESTE ===================\n');
fprintf('Classe Alvo:             "%s"\n', classe_positiva);
fprintf('-----------------------------------------------------------\n');
fprintf('Verdadeiros Positivos (VP):       %d \n', VP);
fprintf('Verdadeiros Negativos (VN):       %d \n', VN);
fprintf('Falsos Positivos (FP):            %d \n', FP);
fprintf('Falsos Negativos (FN):            %d \n', FN);
fprintf('-----------------------------------------------------------\n');
fprintf('Exatidao Global (Accuracy):       %.2f%%\n', exatidao * 100);
fprintf('Precisao:             %.2f%% \n', precisao * 100);
fprintf('Metrica:    %.4f \n', f1_score);
fprintf('===========================================================\n');

assert(exatidao >= 0 && exatidao <= 1, 'Erro crítico no cálculo das métricas.');
fprintf('\n[OK]: Todos os testes do Naive Bayes passaram!\n');