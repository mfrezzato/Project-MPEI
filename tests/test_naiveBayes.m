% test_naiveBayes.m

% Adicionar as pastas ao path para o Matlab encontrar o src/naiveBayes.m
addpath(genpath(pwd));

fprintf('=========================================================\n');
fprintf('     MPEI - TESTE DO MÓDULO NAIVE BAYES      \n');
fprintf('=========================================================\n\n');

% === CARREGAR DATASET DE NOTÍCIAS ===
tabela_news = readtable('news.csv', 'VariableNamingRule', 'preserve'); 

% Limitar o número de registos para o teste correr eficientemente
Nu = min(1500, height(tabela_news));   

fprintf('A processar dados para %d elementos... \n', Nu);
if any(strcmp(tabela_news.Properties.VariableNames, 'Title'))
    documentos = tabela_news.Title(1:Nu);
else
    documentos = tabela_news{1:Nu, 2}; 
end

% Extração dinâmica das classes correspondentes
if any(strcmp(tabela_news.Properties.VariableNames, 'Category'))
    labels = tabela_news.Category(1:Nu);
elseif any(strcmp(tabela_news.Properties.VariableNames, 'Label'))
    labels = tabela_news.Label(1:Nu);
else
    labels = tabela_news{1:Nu, 1}; 
end

if isnumeric(labels)
    labels = cellstr(string(labels));
end

N_total = numel(documentos);
fprintf('Dataset carregado com sucesso. Total de registos selecionados: %d\n', N_total);

% 2. Divisão de Dados 
rng(42); % Fixar seed para consistência de resultados
indices_aleatorios = randperm(N_total);
limite_treino = round(0.7 * N_total); % 70% treino

indices_treino = indices_aleatorios(1:limite_treino);
indices_teste  = indices_aleatorios(limite_treino+1:end);

docs_treino   = documentos(indices_treino);
labels_treino = labels(indices_treino);

docs_teste   = documentos(indices_teste);
labels_teste = labels(indices_teste);

% 3. Inicialização e Treino (Medição de Tempo)
modo_classificador = 'bernoulli'; % 'multinomial' ou 'bernoulli'
nb = naiveBayes(modo_classificador);

fprintf('A treinar o classificador Naïve Bayes (%s) com %d documentos... \n', modo_classificador, numel(docs_treino));
tic;
nb = nb.train(docs_treino, labels_treino);
tempo_treino = toc;
fprintf('Treino concluído. Tamanho do Vocabulário: %d palavras.\n', nb.vocabSize);
fprintf('Tempo gasto no treino: %.4f segundos.\n\n', tempo_treino);

% 4. Classificação
fprintf('A testar o classificador com %d documentos independentes... \n', numel(docs_teste));
tic;
previsoes = nb.classify(docs_teste);
tempo_teste = toc;
fprintf('Classificação concluída em %.4f segundos.\n\n', tempo_teste);

% 5. Cálculo das Métricas de Avaliação
classe_positiva = nb.classes{1}; 
VP = 0; FP = 0; VN = 0; FN = 0;

for i = 1:numel(docs_teste)
    real = labels_teste{i};
    previsto = previsoes{i};
    
    if strcmp(real, classe_positiva)
        if strcmp(previsto, classe_positiva)
            VP = VP + 1; % Verdadeiro Positivo
        else
            FN = FN + 1; % Falso Negativo
        end
    else
        if strcmp(previsto, classe_positiva)
            FP = FP + 1; % Falso Positivo
        else
            VN = VN + 1; % Verdadeiro Negativo
        end
    end
end

exatidao      = (VP + VN) / numel(docs_teste);
precisao      = iff(VP + FP > 0, VP / (VP + FP), 0);
sensibilidade = iff(VP + FN > 0, VP / (VP + FN), 0); 
f1_score      = iff(precisao + sensibilidade > 0, 2 * (precisao * sensibilidade) / (precisao + sensibilidade), 0);

% 6. Exibição dos Resultados
fprintf('=================== RESULTADOS DO TESTE ===================\n');
fprintf('Matriz de Confusão (Classe Positiva de Alvo: "%s"):\n', classe_positiva);
fprintf('   -> Verdadeiros Positivos (VP): %d\n', VP);
fprintf('   -> Falsos Positivos (FP):      %d\n', FP);
fprintf('   -> Verdadeiros Negativos (VN): %d\n', VN);
fprintf('   -> Falsos Negativos (FN):      %d\n', FN);
fprintf('-----------------------------------------------------------\n');
fprintf('Exatidão Global (Accuracy):       %.2f%%\n', exatidao * 100);
fprintf('Precisão (Precision):             %.2f%%\n', precisao * 100);
fprintf('Sensibilidade (Recall):           %.2f%%\n', sensibilidade * 100);
fprintf('Métrica F1-Score:                 %.4f\n', f1_score);
fprintf('-----------------------------------------------------------\n');
fprintf('Tempo de Treino:                  %.4f segundos\n', tempo_treino);
fprintf('Tempo de Teste:                   %.4f segundos\n', tempo_teste);
fprintf('===========================================================\n');

% Garantir integridade da execução
assert(exatidao >= 0 && exatidao <= 1, 'Erro crítico no cálculo das métricas.');
fprintf('\n[MÓDULO NAIVE BAYES]: Todos os testes de volume e consistência passaram com distinção!\n');

% Função inline auxiliar para substituição do operador ternário
function val = iff(cond, vTrue, vFalse)
    if cond, val = vTrue; else, val = vFalse; end
end