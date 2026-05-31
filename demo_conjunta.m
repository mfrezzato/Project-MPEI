% demo_conjunta.m

clear; clc;
fprintf('===================================================================\n');
fprintf('     MPEI - DEMONSTRAÇÃO CONJUNTA: FEED DE NOTÍCIAS INTELIGENTE    \n');
fprintf('===================================================================\n\n');

% Adicionar pastas ao path
addpath(genpath(pwd));

% =========================================================================
% STEP 1: CARREGAMENTO DOS DADOS E TREINO DO SISTEMA
% =========================================================================
fprintf('>>> [Fase de Inicialização] A carregar a base de dados de notícias...\n');
opts = detectImportOptions('news.csv', 'VariableNamingRule', 'preserve');
tabela_news = readtable('news.csv', opts);

% Mapeamento dos índices de classe textuais para maior clareza na demonstração
% Tipicamente: 1-Mundo, 2-Desporto, 3-Economia, 4-Sci/Tech
classMap = containers.Map({'1','2','3','4'}, {'Mundo', 'Desporto', 'Economia', 'Ciência/Tecnologia'});

% Usar os primeiros 1500 registos para alimentar o conhecimento do sistema
N_base = min(1500, height(tabela_news));
titulos_base = tabela_news.Title(1:N_base);
classes_base = tabela_news.('Class Index')(1:N_base);
if isnumeric(classes_base), classes_base = cellstr(string(classes_base)); end

fprintf('A treinar o classificador Naïve Bayes com %d notícias...\n', N_base);
nb_sistema = naiveBayes('multinomial');
nb_sistema = nb_sistema.train(titulos_base, classes_base);

fprintf('A indexar assinaturas MinHash para deteção de similaridade textual...\n');
k_hashes = 100;
mh_sistema = minHash(k_hashes);
Signatures_base = zeros(k_hashes, N_base);

for n = 1:N_base
    shingles = mh_sistema.createShingles(char(titulos_base{n}), 3, 'chars');
    Signatures_base(:, n) = mh_sistema.getSignature(shingles);
end

fprintf('A inicializar o Bloom Filter para histórico de leitura do utilizador...\n');
% Configura o filtro para assumir 500 inserções com taxa de FP de 1%
bf_historico = bloomFilter(500, 0.01, 'classic');

% AJUSTE: Comentário agora alinhado com o código (3 notícias)
% Simular que o utilizador JÁ LEU as primeiras 3 notícias da base de dados
fprintf('-> Simulação: O utilizador acabou de ler as primeiras 3 notícias da base de dados.\n');
for idx_lido = 1:3
    bf_historico = bf_historico.insert(titulos_base{idx_lido});
end
fprintf('[OK]: Sistema inicializado com sucesso e pronto para processar o Feed!\n\n');


% =========================================================================
% STEP 2: SIMULAÇÃO DO PIPELINE DO FEED EM TEMPO REAL
% =========================================================================
fprintf('===================================================================\n');
fprintf('               PROCESSAMENTO DE NOTÍCIAS RECEBIDAS                 \n');
fprintf('===================================================================\n');

% Vamos extrair algumas notícias da frente do dataset para simular que estão a chegar agora
noticias_teste = { ...
    tabela_news.Title{1}, ...                       % 1. Uma que ele JÁ LEU (Deve ser bloqueada pelo Bloom)
    'Cristiano Ronaldo scores an amazing hat-trick', ... % 2. Uma nova de Desporto (Nunca vista)
    tabela_news.Title{50} ...                      % 3. Uma existente não lida (Para testar o MinHash)
};

limiar_similaridade = 0.45; % Limiar de Jaccard para considerar notícias parentes

for t = 1:numel(noticias_teste)
    noticia_atual = noticias_teste{t};
    fprintf('\n[Nova Notícia] Título: "%s"\n', noticia_atual);
    fprintf('-------------------------------------------------------------------\n');
    
    % --- FILTRO 1: BLOOM FILTER (Controlo de Duplicados/Lidos) ---
    if bf_historico.lookup(noticia_atual)
        fprintf('=> [BLOOM FILTER]: [BLOQUEADA] Esta notícia exata já foi lida pelo utilizador!\n');
        fprintf('   Ação: Descartada do feed para evitar redundância.\n');
        continue; 
    else
        fprintf('=> [BLOOM FILTER]: [PASSOU] Notícia inédita para o utilizador.\n');
    end
    
    % --- FILTRO 2: NAÏVE BAYES (Classificação Temática) ---
    classe_prevista_id = nb_sistema.classify({noticia_atual});
    nome_classe = classMap(classe_prevista_id{1});
    
    % Obter probabilidades para enriquecer a interface
    probs_struct = nb_sistema.probability(noticia_atual);
    [max_prob, ~] = max(probs_struct.probabilities);
    
    fprintf('=> [NAÏVE BAYES]: [CATEGORIZADA] Secção do Feed: -> ** %s ** (Confiança: %.2f%%)\n', ...
        nome_classe, max_prob * 100);
    
    % --- FILTRO 3: MINHASH (Deteção de Artigos Similares / Recomendações) ---
    shingles_atual = mh_sistema.createShingles(noticia_atual, 3, 'chars');
    signature_atual = mh_sistema.getSignature(shingles_atual);
    
    % CORREÇÃO CRÍTICA: Vetorização completa em bloco (Sem loops!)
    % Compara a assinatura atual com as 1500 colunas da base de dados simultaneamente
    all_sims = sum(signature_atual == Signatures_base, 1) / k_hashes;
    [melhor_similaridade, index_melhor_similar] = max(all_sims);
    dist_melhor = 1 - melhor_similaridade;
    
    if dist_melhor < limiar_similaridade
        fprintf('=> [MINHASH]: [ALERTA DE SIMILARIDADE] Encontrámos um artigo parente na base de dados!\n');
        fprintf('   Artigo Correspondente: "%s"\n', titulos_base{index_melhor_similar});
        fprintf('   Proximidade Estimada (Jaccard): %.2f%%\n', melhor_similaridade * 100);
        fprintf('   Ação: Agrupando os artigos sob o mesmo tópico dinâmico.\n');
    else
        fprintf('=> [MINHASH]: [CONTEÚDO ÚNICO] Nenhum artigo altamente semelhante encontrado na base.\n');
    end
    
    % Pós-processamento: Assinalar que a notícia foi exibida e passa a constar no histórico
    bf_historico = bf_historico.insert(noticia_atual);
    fprintf('=> [FEED]: Artigo publicado com sucesso no vosso painel personalizado.\n');
end
fprintf('\n===================================================================\n');
fprintf('     [FIM DA DEMONSTRAÇÃO]: Todos os componentes operaram em harmonia! \n');
fprintf('===================================================================\n');